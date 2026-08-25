"""Verification tests for chat_parser + repair_audit on synthetic samples.
Run from code/: python -m tests.test_parser  (or pytest tests/)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chat_parser import parse_cha_file, utterance_feature_row  # noqa: E402
from repair_audit import audit_document  # noqa: E402

DATA = Path(__file__).parent / "data"


def test_dementia_sample():
    doc = parse_cha_file(DATA / "dementia_eng" / "sample1.cha")
    assert doc.languages == ["eng"]
    assert set(doc.participants) == {"PAR", "INV"}
    assert len(doc.utterances) == 8

    u1 = doc.utterances[1]  # first PAR utterance (continuation folded)
    assert u1.speaker == "PAR"
    assert len(u1.filled_pauses) == 2          # &-um, &-uh
    assert u1.retracings == 1                  # [//]
    assert u1.untimed_pauses == 1              # (.)
    assert u1.start_ms == 3400 and u1.end_ms == 11250
    assert "cookies" in u1.clean_tokens

    u2 = doc.utterances[2]                     # "and the the [/] the sink is +..."
    assert u2.repetitions == 1                 # [/]
    assert u2.terminator_flags["trailing_off"]

    u4 = doc.utterances[4]                     # timed pause + xxx
    assert u4.timed_pause_total_s == 2.5
    assert u4.unintelligible == 1
    assert "mor" in u4.dependent_tiers

    u6 = doc.utterances[6]
    assert u6.terminator_flags["self_interruption"]

    u7 = doc.utterances[7]
    assert "because" in u7.clean_tokens        # (be)cause expansion

    inv_q = doc.utterances[3]
    assert inv_q.is_question and inv_q.speaker == "INV"

    row = utterance_feature_row(doc, u1, corpus="Pitt")
    assert row["filled_pauses"] == 2 and row["duration_s"] == 7.85
    assert row["speech_rate_tok_per_s"] > 0

    audit = audit_document(doc, corpus="Pitt")
    assert audit["n_inv_utts"] == 3
    assert audit["inv_questions"] == 2
    assert audit["candidate_repair_sequences"] >= 1   # trailing-off -> INV ?
    assert audit["timestamp_coverage"] == 1.0


def test_dialogue_sample():
    doc = parse_cha_file(DATA / "dialogue_deu" / "sample2.cha")
    audit = audit_document(doc, corpus="SampleDeu")
    assert audit["language"] == "deu"
    assert audit["par_ntri"] >= 1              # "was ?"
    assert audit["inv_reasks"] >= 1            # repeated INV question
    assert audit["timestamp_coverage"] == 0.0  # no bullets -> alignment needed
    u = doc.utterances[3]
    assert len(u.filled_pauses) == 1 and u.repetitions == 1


if __name__ == "__main__":
    test_dementia_sample()
    test_dialogue_sample()
    print("ALL TESTS PASSED")
