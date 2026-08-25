"""Generate a synthetic multi-corpus dataset exercising the full pipeline:
3 languages x 2 populations, longitudinal sessions, severity-graded disfluency.
Verification only — numbers from this corpus mean nothing scientifically.
Run: python tests/make_synth_corpus.py [out_dir]"""
import random
import sys
from pathlib import Path

B = "\x15"
WORDS = {
    "eng": "the boy is taking a cookie from the jar while water runs over".split(),
    "deu": "der Junge nimmt einen Keks aus der Dose während Wasser überläuft".split(),
    "zho": "男孩 从 罐子 里 拿 饼干 水 从 水槽 里 溢出".split(),
}
GROUPS = {"dementia": ["Control", "MCI", "ProbableAD"], "fluency": ["Control", "AWS"]}
SEVERITY = {"Control": 0.03, "MCI": 0.10, "AWS": 0.15, "ProbableAD": 0.22}


def utterance(lang: str, sev: float, rng: random.Random, t0: int) -> tuple[str, int]:
    toks = rng.choices(WORDS[lang], k=rng.randint(4, 12))
    outp = []
    for w in toks:
        r = rng.random()
        if r < sev * 0.4:
            outp += ["&-um" if lang == "eng" else "&-äh" if lang == "deu" else "&-嗯"]
        if r < sev * 0.3:
            outp += [w, "[/]"]
        elif r < sev * 0.45:
            outp += [w, "[//]"]
        if rng.random() < sev * 0.5:
            outp.append("(.)" if rng.random() < 0.7 else f"({rng.uniform(0.5, 3.0):.1f})")
        outp.append(w)
    term = "+..." if rng.random() < sev * 0.4 else "."
    dur = int((len(toks) / max(0.5, 2.5 - 4 * sev)) * 1000)
    text = " ".join(outp) + f" {term} {B}{t0}_{t0 + dur}{B}"
    return text, t0 + dur + rng.randint(200, 900)


def make_file(path: Path, lang: str, corpus: str, sid: str, session: int,
              group: str, rng: random.Random):
    sev = SEVERITY[group]
    lines = ["@UTF8", "@Begin", f"@Languages:\t{lang}",
             "@Participants:\tPAR Participant, INV Investigator",
             f"@ID:\t{lang}|{corpus}|PAR|{rng.randint(55, 85)};|"
             f"{rng.choice(['male', 'female'])}|{group}||Participant|||"]
    t = 0
    for _ in range(rng.randint(8, 14)):
        if rng.random() < 0.3:
            q, t = utterance(lang, 0.02, rng, t)
            lines.append(f"*INV:\ttell me more ? {B}{t}_{t + 1200}{B}")
            t += 1500
        u, t = utterance(lang, sev, rng, t)
        lines.append(f"*PAR:\t{u}")
        # comprehension trouble -> NTRI + INV re-ask (repair sequence)
        if rng.random() < sev * 0.6:
            lines.append(f"*PAR:\twhat ? {B}{t}_{t + 600}{B}")
            lines.append(f"*INV:\ttell me more ? {B}{t + 700}_{t + 1900}{B}")
            t += 2200
    lines.append("@End")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(out_dir: str):
    rng = random.Random(7)
    spec = [  # corpus, lang, population, n_speakers, n_sessions
        ("Pitt", "eng", "dementia", 18, 2),
        ("German", "deu", "dementia", 12, 1),
        ("Lu_Mandarin", "zho", "dementia", 12, 1),
        ("FluencyBank", "eng", "fluency", 12, 1),
    ]
    root = Path(out_dir)
    for corpus, lang, pop, n_spk, n_sess in spec:
        for i in range(n_spk):
            groups = GROUPS[pop]
            group = groups[i % len(groups)]
            for s in range(n_sess):
                make_file(root / corpus / f"{i + 1:03d}-{s}.cha",
                          lang, corpus, f"{i + 1:03d}", s, group, rng)
    print(f"Wrote synthetic corpus -> {root}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/synth_data")
