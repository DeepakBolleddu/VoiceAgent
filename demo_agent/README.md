# Live demo — communicative-difficulty sensing + adaptation

Push-to-talk voice agent: your prof speaks English; the trained estimator
scores each turn's communicative difficulty (calibrated as a percentile of
clinical speech); a tiered policy adapts the agent's replies — normal → slower
→ simplified → teach-back comprehension check. Saves a session trajectory
chart on exit.

## One-time setup

**On the HPC** (packages model + calibration):
```bash
cd ~/VoiceAgent/benchmark
python scripts/export_demo_model.py --config configs/hpc.yaml --run probe
```

**On your Mac:**
```bash
scp -r u8023272@hpc-login-prd-t1:~/VoiceAgent/benchmark/demo_model \
      ~/Documents/PhD/Research/VoiceAgent/demo_agent/
cd ~/Documents/PhD/Research/VoiceAgent/demo_agent
pip install torch transformers soundfile sounddevice matplotlib numpy
pip install openai-whisper        # optional: shows a live transcript
```

## Run

```bash
python live_demo.py               # conversational demo (Whisper ASR on by default)
```

The agent greets you, replies to WHAT you say, adapts HOW it says it to the
sensed difficulty, and writes `session_report.md` + trajectory chart on Ctrl-C.

**Reply brain — pick one before the meeting (auto-detected in this order):**

1. `export ANTHROPIC_API_KEY=sk-...` → full conversational replies (Claude Haiku) — best demo.
2. Ollama running locally (`ollama serve`, any chat model) → local LLM replies.
3. Neither → reflective mode: mirrors your words + tier-appropriate follow-up
   (still content-aware, just simpler).

Do a full rehearsal the day before: mic permissions, model download, one
fluent + one hesitant turn, and confirm which brain mode prints at startup.

## Demo script for the meeting (2 minutes)

1. Speak a fluent sentence → green meter, normal reply.
2. Speak with deliberate trouble ("so... um... the— the thing I wanted... it's
   about the... you know...") → meter climbs, agent slows and simplifies.
3. Push it further (long pauses, trail off) → TEACH_BACK tier: the agent asks
   for the information back — the clinical safety behavior.
4. Ctrl-C → show `session_trajectory.png`: difficulty as a state, moving
   within one conversation.

## Say this out loud (framing)

Trained on multilingual clinical speech with weak transcript-derived labels;
healthy speakers are out-of-domain; thresholds are demonstration defaults, not
clinically validated. This demonstrates the **sensing→adaptation loop** — the
research contribution — not a deployable medical device. Translation (the
cascaded ASR→MT→TTS agent) is the Phase-2 demonstrator.

## Voice quality (recommended)

```bash
pip install edge-tts
```
With edge-tts installed the agent speaks in a natural neural voice with
per-tier **rate and pitch** adaptation (lower/calmer as difficulty rises — the
prosody layer). Needs internet. Without it, falls back to macOS `say`.

## Baseline calibration (why the demo starts with two warm-up questions)

Healthy speakers score high on the clinical-speech scale (domain shift: age,
accent, microphone). So the agent first learns YOUR fluent baseline from two
relaxed turns, then scores every turn **relative to your own normal** — which
is also the construct itself (difficulty is a state: deviation from one's own
baseline). Skip with `--no-calibrate`. If the demo speaker changes, restart so
it recalibrates. For accented English use `--whisper-model medium.en`.

## Known limits (say these if asked)

- Whisper normalizes speech: light grammar "fixing" and dropped stutters are an
  ASR property, not a sensing error — **difficulty is scored from audio, never
  from the transcript**. Verbatim disfluency ASR (CrisperWhisper-class) is the
  Phase-2 fix. `--whisper-model medium.en` reduces word errors if your Mac
  handles it.
- Pause sensing counts only silences ≥300 ms (word gaps excluded), and reports
  response latency (silence before you start speaking) separately.

## Troubleshooting

- No mic input: System Settings → Privacy → Microphone → allow your terminal.
- `say` not speaking: use `--no-voice` (text-only replies still shown).
- Slow scoring on CPU: keep turns under ~15 s; scoring takes ~1–3 s per turn.
