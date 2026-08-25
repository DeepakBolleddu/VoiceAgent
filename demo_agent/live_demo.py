"""
live_demo.py v2 — Conversational communicative-difficulty agent (laptop demo).

The loop:  greet -> [push-to-talk] -> Whisper ASR -> difficulty score
(calibrated percentile) -> adaptation tier -> content-aware reply whose
REGISTER adapts to the tier -> spoken reply -> ... -> session report.

Reply "brain" (auto-detected, best available):
  1. Claude API      — set ANTHROPIC_API_KEY in the environment
  2. Ollama (local)  — `ollama serve` with any chat model (e.g. llama3.2)
  3. Reflective mode — no LLM: mirrors your words + tier-appropriate follow-up

Setup:
  pip install torch transformers soundfile sounddevice matplotlib numpy openai-whisper
  (demo_model/ exported from the HPC — see README)
Run:
  python live_demo.py            # conversational demo
  python live_demo.py --no-voice # silent (text only)

Framing: trained on clinical speech with weak labels; healthy speakers are
out-of-domain; a demonstrator of the sensing->adaptation loop.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

# --- adaptation tiers (plan §7.2) --------------------------------------------
# Thresholds apply to the RELATIVE score: how far this turn rises above the
# speaker's own calibrated baseline (state = deviation from one's own normal).
TIERS = [
    (0.00, "FLOW"),
    (0.30, "SLOW"),
    (0.60, "SIMPLIFY"),
    (0.90, "TEACH_BACK"),
]
SAY_RATES = {"FLOW": 185, "SLOW": 150, "SIMPLIFY": 135, "TEACH_BACK": 130}

STYLE = {
    "FLOW": "Reply naturally and warmly in 1-2 sentences. Continue the topic; "
            "you may ask one open-ended follow-up question.",
    "SLOW": "The speaker showed mild communicative difficulty. Reply in short, "
            "clear sentences. One idea per sentence. Stay on their topic. End "
            "with one gentle, easy question (not open-ended).",
    "SIMPLIFY": "The speaker showed clear communicative difficulty. Use very "
                "simple words. Maximum two short sentences. Address only the "
                "main thing they said. Ask ONE binary or forced-choice question "
                "(yes/no, or 'X or Y?') — never an open-ended one.",
    "TEACH_BACK": "The speaker showed severe communicative difficulty. First, "
                  "state in one very simple sentence what you understood from "
                  "them. Then ask them to tell you, in their own words, what "
                  "you both just said — to make sure you understood each other.",
    "REGROUND": "Difficulty has stayed high for several turns. Do NOT ask any "
                "question. Speak one or two short, calm, reassuring sentences. "
                "Validate their feeling. ONLY IF they actually shared a "
                "comforting personal detail EARLIER IN THIS CONVERSATION, you "
                "may gently mention it; NEVER invent or assume one. Otherwise "
                "simply reassure them and let them rest.",
}
SYSTEM = ("You are a friendly spoken conversation partner in a live demo of a "
          "clinical communication-support agent. Keep every reply brief and "
          "speakable (this is voice, not text). Never give medical advice or "
          "diagnosis; defer clinical questions to medical staff. "
          "VALIDATION RULE: never correct, contradict, or quiz the speaker "
          "about facts, places, or times — acknowledge the feeling or memory "
          "first, then gently continue. "
          "ANCHOR MEMORY: remember personal details they share (people, "
          "places, foods, memories) and naturally reuse them in later turns so "
          "they never have to re-explain. "
          "Adapt your register exactly as instructed per turn.")

GREETING = ("Hi! I'm your communication assistant. We can chat about anything "
            "you like. As we talk, I'll pay attention to how the conversation "
            "is flowing, and I'll adapt to you.")

CALIB_PROMPTS = [
    "First, let's get to know your voice. Please tell me, in a relaxed way, "
    "what you did this morning.",
    "Lovely. One more — tell me about something you enjoy doing.",
]
CALIB_DONE = "Thank you, I've got a feel for your voice now. So — how is your day going?"


def tier_for(pct: float) -> str:
    name = TIERS[0][1]
    for th, nm in TIERS:
        if pct >= th:
            name = nm
    return name


# --- reply brains -------------------------------------------------------------
def _post_json(url: str, payload: dict, headers: dict, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class Brain:
    """Best-available response generator with graceful fallback."""

    def __init__(self):
        self.mode = "reflective"
        self.history: list[dict] = []
        if os.environ.get("ANTHROPIC_API_KEY"):
            self.mode = "claude"
        elif os.environ.get("GEMINI_API_KEY"):
            self.mode = "gemini"
            self.gemini_model = os.environ.get("GEMINI_MODEL") or self._pick_gemini()
        else:
            try:
                with urllib.request.urlopen("http://localhost:11434/api/tags",
                                            timeout=2) as r:
                    models = json.loads(r.read()).get("models", [])
                self.mode = "ollama"
                self.ollama_model = models[0]["name"] if models else "llama3.2"
            except Exception:
                pass
        print(f"[brain] reply mode: {self.mode}"
              + ("" if self.mode != "reflective" else
                 "  (set ANTHROPIC_API_KEY / GEMINI_API_KEY or run Ollama "
                 "for full conversation)"))

    def reply(self, user_text: str, tier: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        try:
            if self.mode == "claude":
                out = self._claude(tier)
            elif self.mode == "gemini":
                out = self._gemini(tier)
            elif self.mode == "ollama":
                out = self._ollama(tier)
            else:
                out = self._reflect(user_text, tier)
        except Exception as e:
            # Gemini rate-limited/missing? rotate to the next candidate model
            # so the NEXT turn recovers automatically.
            if (self.mode == "gemini"
                    and len(getattr(self, "gemini_candidates", [])) > 1):
                self.gemini_candidates.pop(0)
                self.gemini_model = self.gemini_candidates[0]
                print(f"  (brain error: {e} — switching to "
                      f"{self.gemini_model} for next turn)")
            else:
                print(f"  (brain error: {e} — falling back to reflective)")
            out = self._reflect(user_text, tier)
        self.history.append({"role": "assistant", "content": out})
        return out

    def _claude(self, tier: str) -> str:
        msgs = self.history[-10:]
        msgs = msgs[:-1] + [{"role": "user",
                             "content": f"{msgs[-1]['content']}\n\n"
                                        f"[REGISTER INSTRUCTION: {STYLE[tier]}]"}]
        r = _post_json("https://api.anthropic.com/v1/messages",
                       {"model": "claude-haiku-4-5", "max_tokens": 150,
                        "system": SYSTEM, "messages": msgs},
                       {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                        "anthropic-version": "2023-06-01"})
        return r["content"][0]["text"].strip()

    def _pick_gemini(self) -> str:
        """Ask the key which models it can use. Prefer STABLE flash models —
        preview/experimental ones have tiny rate quotas (instant 429s). Keep a
        ranked candidate list so we can rotate away from a rate-limited model."""
        candidates = ["gemini-2.0-flash"]
        try:
            req = urllib.request.Request(
                "https://generativelanguage.googleapis.com/v1beta/models?pageSize=100",
                headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
            with urllib.request.urlopen(req, timeout=10) as r:
                models = json.loads(r.read()).get("models", [])
            usable = [m["name"].split("/")[-1] for m in models
                      if "generateContent" in m.get("supportedGenerationMethods", [])]
            flash = [m for m in usable if "flash" in m
                     and not any(x in m for x in ("image", "tts", "live", "audio"))]
            stable = sorted([m for m in flash
                             if not any(x in m for x in ("preview", "exp"))],
                            reverse=True)
            preview = sorted([m for m in flash if m not in stable], reverse=True)
            candidates = (stable + preview) or usable or candidates
        except Exception as e:
            print(f"[brain] gemini model discovery failed ({e}); using defaults")
        self.gemini_candidates = candidates
        print(f"[brain] gemini model auto-selected: {candidates[0]}"
              + (f"  (fallbacks: {', '.join(candidates[1:3])})"
                 if len(candidates) > 1 else ""))
        return candidates[0]

    def _gemini(self, tier: str) -> str:
        # Gemini REST (no SDK needed). Roles: user / model.
        contents = []
        for m in self.history[-10:-1]:
            contents.append({"role": "user" if m["role"] == "user" else "model",
                             "parts": [{"text": m["content"]}]})
        contents.append({"role": "user",
                         "parts": [{"text": f"{self.history[-1]['content']}\n\n"
                                            f"[REGISTER INSTRUCTION: {STYLE[tier]}]"}]})
        r = _post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.gemini_model}:generateContent",
            {"system_instruction": {"parts": [{"text": SYSTEM}]},
             "contents": contents,
             "generationConfig": {"maxOutputTokens": 150}},
            {"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        return r["candidates"][0]["content"]["parts"][0]["text"].strip()

    def _ollama(self, tier: str) -> str:
        msgs = ([{"role": "system", "content": SYSTEM}]
                + self.history[-10:-1]
                + [{"role": "user",
                    "content": f"{self.history[-1]['content']}\n\n"
                               f"[REGISTER INSTRUCTION: {STYLE[tier]}]"}])
        r = _post_json("http://localhost:11434/api/chat",
                       {"model": self.ollama_model, "messages": msgs,
                        "stream": False}, {}, timeout=60)
        return r["message"]["content"].strip()

    def _reflect(self, text: str, tier: str) -> str:
        """No-LLM fallback: mirror content + tier-appropriate follow-up."""
        gist = text.strip().rstrip(".!?")
        gist = gist if len(gist) < 90 else gist[:87] + "..."
        if tier == "REGROUND":   # calm, no question, no demand
            return ("That's okay. Take your time — there is no hurry. "
                    "I'm right here with you.")
        # If they gave a short answer (e.g. replied to our yes/no question),
        # acknowledge and MOVE ON to a fresh, easy topic instead of looping.
        if len(text.split()) <= 5:
            topics = ["What did you have for breakfast today?",
                      "What do you enjoy doing in the afternoons?",
                      "Tell me about someone in your family?",
                      "What was the weather like today?",
                      "What food reminds you of home?"]
            self._topic_i = getattr(self, "_topic_i", -1) + 1
            nxt = topics[self._topic_i % len(topics)]
            if tier in ("SIMPLIFY", "TEACH_BACK"):
                nxt = {"What did you have for breakfast today?":
                           "Did you eat breakfast today — yes or no?",
                       "What do you enjoy doing in the afternoons?":
                           "Do you like mornings or afternoons more?",
                       "Tell me about someone in your family?":
                           "Do you have brothers or sisters?",
                       "What was the weather like today?":
                           "Is it sunny or cloudy today?",
                       "What food reminds you of home?":
                           "Do you like sweet food or spicy food?"}[nxt]
            return f"Thank you. {nxt}"
        follow = {
            "FLOW": "That's interesting — tell me more?",
            "SLOW": "I want to follow you. What is the main thing?",
            "SIMPLIFY": "Let's go step by step. Is that the main thing — yes or no?",
            "TEACH_BACK": "Can you tell me again, in your own words, so I'm "
                          "sure I understood you?",
        }.get(tier, "Please go on.")
        lead = {"FLOW": "I hear you", "SLOW": "Okay, slowly now",
                "SIMPLIFY": "Okay", "TEACH_BACK": "Let me check"}.get(tier, "Okay")
        return f'{lead} — you said: "{gist}". {follow}'


# --- sensing -------------------------------------------------------------------
class Difficulty:
    def __init__(self, model_dir: Path):
        import torch
        from transformers import AutoFeatureExtractor, AutoModel
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark"))
        from vabench.models.estimator import DifficultyEstimator

        meta = json.loads((model_dir / "meta.json").read_text())
        self.layer = meta["layer"]
        self.quant = np.asarray(meta["score_quantiles"])
        print(f"[init] loading {meta['ssl_model']}")
        self.fe = AutoFeatureExtractor.from_pretrained(meta["ssl_model"])
        self.ssl = AutoModel.from_pretrained(meta["ssl_model"],
                                             output_hidden_states=True).eval()
        self.head = DifficultyEstimator(
            in_dim=meta["in_dim"], n_langs=meta["n_langs"],
            n_pops=meta["n_pops"], n_spks=meta["n_spks"])
        self.head.load_state_dict(torch.load(model_dir / "model.pt",
                                             map_location="cpu"))
        self.head.eval()
        self.torch = torch

    def score(self, wav: np.ndarray, sr: int = 16000) -> dict:
        with self.torch.no_grad():
            inp = self.fe(wav, sampling_rate=sr, return_tensors="pt")
            h = self.ssl(**inp).hidden_states[self.layer].mean(dim=1)
            raw = float(self.head(h)["difficulty"].item())
        model_pct = float(np.searchsorted(self.quant, raw) / (len(self.quant) - 1))
        ac = acoustic_features(wav, sr)
        # Pause evidence (absolute temporal markers) is fused LATER, in
        # baseline-relative space — otherwise a high personal baseline can sit
        # above the pause cap and hesitation evidence never registers.
        pause_pct = min(1.0, 0.6 * min(ac["longest_pause_s"] / 2.5, 1.0)
                        + 0.4 * min(ac["pause_ratio"] / 0.5, 1.0))
        ratio_ok = ac["pause_ratio"] >= 0.35 and ac.get("core_s", 0) >= 3.0
        substantial = (ac["longest_pause_s"] >= 1.2 or ratio_ok
                       or ac["response_latency_s"] >= 2.0)
        return {"pct": model_pct, "model_pct": model_pct,
                "pause_pct": pause_pct, "substantial": substantial, **ac}


MIN_PAUSE_S = 0.30   # silences shorter than this are word gaps, not pauses


def acoustic_features(wav: np.ndarray, sr: int = 16000) -> dict:
    """Interpretable temporal markers (spec layer 1).
    - pause_ratio: fraction of the spoken core spent in silences >= 300 ms
      (word gaps excluded — fluent speech should score ~0)
    - longest_pause_s: longest internal silence
    - response_latency_s: silence between record start and first speech
      (word-finding latency after the agent's question)"""
    zeros = {"pause_ratio": 0.0, "longest_pause_s": 0.0, "response_latency_s": 0.0}
    frame = int(0.03 * sr)
    n = len(wav) // frame
    if n < 5:
        return zeros
    e = np.array([float(np.sqrt(np.mean(wav[i*frame:(i+1)*frame]**2)))
                  for i in range(n)])
    noise = np.percentile(e, 10)
    speech = np.percentile(e, 90)
    thr = noise + 0.15 * max(speech - noise, 1e-6)
    silent = e < thr
    idx = np.where(~silent)[0]
    if len(idx) < 2:
        return {**zeros, "response_latency_s": round(n * frame / sr, 2)}
    latency = idx[0] * frame / sr
    core = silent[idx[0]:idx[-1] + 1]
    # silent runs within the spoken core
    runs, run = [], 0
    for s in core:
        if s:
            run += 1
        elif run:
            runs.append(run); run = 0
    if run:
        runs.append(run)
    long_runs = [r for r in runs if r * frame / sr >= MIN_PAUSE_S]
    core_s = len(core) * frame / sr
    return {"pause_ratio": round(sum(long_runs) * frame / sr / core_s, 3),
            "longest_pause_s": round((max(runs) if runs else 0) * frame / sr, 2),
            "response_latency_s": round(latency, 2),
            "core_s": round(core_s, 2)}


FILLERS_EN = {"um", "uh", "er", "erm", "uhm", "mm", "hmm", "mhm", "eh"}


def text_disfluencies(text: str) -> dict:
    """Transcript-side markers (reported, not fused): fillers, immediate word
    repetitions (stutters/restarts), and hesitation ellipses from the ASR."""
    toks = [t.lower().strip(".,!?;:\"'") for t in text.split()]
    toks = [t for t in toks if t]
    fillers = sum(1 for t in toks if t in FILLERS_EN)
    repeats = sum(1 for a, b in zip(toks, toks[1:]) if a == b and a not in FILLERS_EN)
    ellipses = text.count("...") + text.count("- ") + text.count("—")
    return {"fillers": fillers, "repeats": repeats, "ellipses": ellipses}


def record_push_to_talk(sr: int = 16000) -> np.ndarray | None:
    import sounddevice as sd
    q: queue.Queue = queue.Queue()
    def cb(indata, frames, t, status):
        q.put(indata.copy())
    try:
        input("\n[Enter] to START speaking...")
        print("  recording — [Enter] to STOP")
        with sd.InputStream(samplerate=sr, channels=1, callback=cb):
            input()
    except EOFError:          # closed stdin must not eat the session report
        return None
    chunks = []
    while not q.empty():
        chunks.append(q.get())
    return (np.concatenate(chunks)[:, 0] if chunks
            else np.zeros(sr // 2, dtype="float32"))


def meter(pct: float, width: int = 40) -> str:
    n = int(pct * width)
    color = "\033[92m" if pct < .5 else "\033[93m" if pct < .75 else "\033[91m"
    return f"{color}{'█'*n}{'░'*(width-n)}\033[0m {pct*100:5.1f}%"


# Neural prosody per tier (spec layer 3): slower AND lower/calmer as
# difficulty rises. Used by edge-tts; `say` fallback gets rate only.
EDGE_PROSODY = {
    "FLOW":       ("+0%",  "+0Hz"),
    "SLOW":       ("-15%", "-5Hz"),
    "SIMPLIFY":   ("-25%", "-10Hz"),
    "TEACH_BACK": ("-28%", "-10Hz"),
    "REGROUND":   ("-30%", "-15Hz"),
}
_EDGE_VOICE = "en-AU-NatashaNeural"     # natural neural voice (needs internet)


def _edge_speak(text: str, tier: str, block: bool) -> bool:
    try:
        import asyncio
        import tempfile
        import edge_tts
    except ImportError:
        return False
    try:
        rate, pitch = EDGE_PROSODY.get(tier, EDGE_PROSODY["SIMPLIFY"])
        f = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        asyncio.run(edge_tts.Communicate(text, _EDGE_VOICE, rate=rate,
                                         pitch=pitch).save(f.name))
        p = subprocess.Popen(["afplay", f.name])
        if block:
            p.wait()
        return True
    except Exception:
        return False


def speak(text: str, tier: str, enabled: bool, block: bool = False):
    if not enabled:
        return
    if _edge_speak(text, tier, block):     # natural neural voice, rate+pitch
        return
    if sys.platform == "darwin":           # fallback: built-in say (rate only)
        p = subprocess.Popen(["say", "-r",
                              str(SAY_RATES.get(tier, SAY_RATES["SIMPLIFY"])),
                              text])
        if block:
            p.wait()


def write_report(turns: list[dict], brain_mode: str, alerts: list[str] | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not turns:
        return
    t0 = turns[0]["t"]
    ts = [(x["t"] - t0) / 60 for x in turns]
    ps = [x["pct"] for x in turns]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(ts, ps, marker="o")
    for th, nm in TIERS[1:]:
        ax.axhline(th, ls="--", lw=.8, color="gray")
        ax.text(ax.get_xlim()[1], th, f" {nm}", va="center", fontsize=8)
    ax.set_xlabel("time (min)"); ax.set_ylabel("difficulty percentile")
    ax.set_ylim(0, 1); ax.set_title("Session — communicative difficulty over time")
    fig.tight_layout(); fig.savefig("session_trajectory.png", dpi=160)

    from collections import Counter
    tiers = Counter(x["tier"] for x in turns)
    esc = sum(1 for x in turns if x["tier"] != "FLOW")
    lines = [
        "# Session report — communicative-difficulty agent (demonstrator)",
        "",
        f"- Turns: **{len(turns)}** · Reply brain: **{brain_mode}**",
        f"- Mean difficulty: **{np.mean(ps)*100:.0f}th percentile** of clinical "
        f"speech · Peak: **{max(ps)*100:.0f}th** ",
        f"- Adaptations triggered: **{esc}/{len(turns)}** turns "
        f"({', '.join(f'{k}×{v}' for k, v in tiers.items())})",
        "- Difficulty varied within the session -> evidence of a *state*, "
        "not a fixed speaker trait.",
    ]
    if alerts:
        lines.append(f"- **⚠ Escalation alerts ({len(alerts)}):** "
                     + "; ".join(alerts))
    lines += [
        "",
        "![trajectory](session_trajectory.png)",
        "",
        "| # | you said | difficulty | pause | longest | tier | agent replied |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, x in enumerate(turns, 1):
        lines.append(f"| {i} | {x['text'][:50]} | {x['pct']*100:.0f}% "
                     f"| {x.get('pause_ratio',0)*100:.0f}% "
                     f"| {x.get('longest_pause_s',0):.1f}s "
                     f"| {x['tier']} | {x['reply'][:50]} |")
    lines += ["", "*Demonstrator only: model trained on clinical speech with "
              "weak labels; healthy speakers are out-of-domain; thresholds are "
              "demo defaults, not clinically validated.*"]
    Path("session_report.md").write_text("\n".join(lines))
    print(f"\n[demo] wrote session_report.md + session_trajectory.png "
          f"({len(turns)} turns)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="demo_model")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--no-asr", action="store_true", help="skip Whisper (meter only)")
    ap.add_argument("--whisper-model", default="small.en",
                    help="use medium.en for much better accent robustness (1.5GB)")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="skip per-speaker baseline enrollment")
    args = ap.parse_args()

    diff = Difficulty(Path(args.model_dir))
    asr = None
    if not args.no_asr:
        import whisper
        print(f"[init] loading Whisper ({args.whisper_model})")
        asr = whisper.load_model(args.whisper_model)
    # Nudge Whisper to keep fillers instead of deleting them (partial fix;
    # verbatim disfluency ASR, e.g. CrisperWhisper, is the Phase-2 answer).
    ASR_PROMPT = ("Transcribe verbatim, keeping hesitations and repeated words, "
                  "for example: um, uh, er, I- I mean, the the thing...")
    brain = Brain()

    print("\n=== Communicative-Difficulty Voice Agent (demonstrator) ===")
    print("Chat naturally. The agent listens, senses difficulty, and adapts.")
    print("Ctrl-C to finish (writes session_report.md).\n")
    print(f"  agent: {GREETING}")
    speak(GREETING, "FLOW", not args.no_voice, block=True)

    # --- per-speaker baseline calibration (state = deviation from YOUR normal;
    # also absorbs accent/mic/channel domain shift) --------------------------
    baseline = None
    if not args.no_calibrate:
        calib_scores = []
        for prompt in CALIB_PROMPTS:
            print(f"\n  agent: {prompt}")
            speak(prompt, "FLOW", not args.no_voice, block=True)
            wav = record_push_to_talk()
            if wav is None or len(wav) / 16000 < 0.4:
                continue
            calib_scores.append(diff.score(wav)["pct"])
        if calib_scores:
            baseline = float(np.mean(calib_scores))
            print(f"\n  [calibrated: your fluent baseline = "
                  f"{baseline*100:.0f}% of the clinical scale — scoring "
                  f"RELATIVE to you from here]")
        print(f"\n  agent: {CALIB_DONE}")
        speak(CALIB_DONE, "FLOW", not args.no_voice, block=True)

    def relative(pct: float) -> float:
        """Rise above the personal baseline, with a minimum headroom floor so
        a very high baseline (e.g. 0.89) still leaves usable dynamic range."""
        if baseline is None:
            return pct
        return min(max((pct - baseline) / max(0.15, 1.0 - baseline), 0.0), 1.0)

    turns: list[dict] = []
    alerts: list[str] = []
    try:
        while True:
            wav = record_push_to_talk()
            if wav is None:
                break
            if len(wav) / 16000 < 0.4:
                print("  (too short, try again)"); continue
            s = diff.score(wav)
            rel_model = relative(s["model_pct"])   # rise above YOUR baseline
            # temporal hesitation evidence is absolute — fuse in relative space
            pct = (max(rel_model, min(s["pause_pct"], 0.85))
                   if s["substantial"] else rel_model)
            tier = tier_for(pct)

            # Escalation matrix (spec layer 4): 3 consecutive high-tier turns
            # -> soft re-grounding + caregiver alert logged.
            recent = [t["tier"] for t in turns[-2:]] + [tier]
            if (len(recent) == 3
                    and all(r in ("SIMPLIFY", "TEACH_BACK") for r in recent)):
                tier = "REGROUND"
                alert = (f"turn {len(turns)+1}: difficulty high for 3 "
                         f"consecutive turns — caregiver/nursing alert logged")
                alerts.append(alert)
                print(f"  \033[91m⚠ ESCALATION: {alert}\033[0m")

            text = ""
            if asr is not None:
                text = asr.transcribe(wav.astype(np.float32), fp16=False,
                                      temperature=0.0,
                                      condition_on_previous_text=False,
                                      initial_prompt=ASR_PROMPT)["text"].strip()
                print(f'  you said: "{text}"')
            td = text_disfluencies(text) if text else {"fillers": 0,
                                                       "repeats": 0, "ellipses": 0}
            marks = [f"abs {s['model_pct']*100:.0f}%"]
            if baseline is not None:
                marks.append(f"baseline {baseline*100:.0f}%")
            if s["substantial"]:
                marks.append("hesitation-evidence")
            if s["response_latency_s"] >= 0.5:
                marks.append(f"latency {s['response_latency_s']:.1f}s")
            if s["longest_pause_s"] >= 0.4:
                marks.append(f"pause {s['longest_pause_s']:.1f}s")
            if s["pause_ratio"] >= 0.1:
                marks.append(f"pausing {s['pause_ratio']*100:.0f}%")
            if td["fillers"]:
                marks.append(f"fillers {td['fillers']}")
            if td["repeats"]:
                marks.append(f"stutter/repeat {td['repeats']}")
            if td["ellipses"]:
                marks.append(f"trail-offs {td['ellipses']}")
            print(f"  difficulty {meter(pct)}   -> tier: {tier}  [{', '.join(marks)}]")
            reply = brain.reply(text or "(unintelligible)", tier)
            print(f"  agent: {reply}")
            speak(reply, tier, not args.no_voice)
            turns.append({"t": time.time(), "text": text, "pct": pct,
                          "tier": tier, "reply": reply,
                          "pause_ratio": s["pause_ratio"],
                          "longest_pause_s": s["longest_pause_s"]})
    except KeyboardInterrupt:
        pass
    write_report(turns, brain.mode, alerts)


if __name__ == "__main__":
    main()
