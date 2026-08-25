"""Generate synthetic .cha files mimicking TalkBank structure, for pipeline
verification only (no real data). Run: python make_samples.py"""
from pathlib import Path

B = "\x15"  # media bullet delimiter

SAMPLE_DEMENTIA = f"""@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant, INV Investigator
@ID:\teng|Pitt|PAR|68;|female|ProbableAD||Participant|||
@Media:\tsample_dementia, audio
*INV:\tokay , tell me everything you see going on in that picture . {B}0_3200{B}
*PAR:\twell there's a &-um a boy on a &-uh stool [//] a ladder (.) taking
\tcookies . {B}3400_11250{B}
*PAR:\tand the the [/] the sink is +... {B}11500_16800{B}
*INV:\twhat about the sink ? {B}17000_18400{B}
*PAR:\tthe water (2.5) the water is xxx over . {B}18600_25100{B}
%mor:\tdet:art|the n|water n|water cop|be&3S adv|over .
*INV:\tthe water is running over ? {B}25300_27000{B}
*PAR:\tyes running over the +//. {B}27200_30500{B}
*PAR:\t&-um the mother is (be)cause she's [//] the mother's drying dishes . {B}30700_37900{B}
@End
"""

SAMPLE_DIALOGUE = f"""@UTF8
@Begin
@Languages:\tdeu
@Participants:\tPAR Participant, INV Investigator
@ID:\tdeu|Sample|PAR|71;|male|Control||Participant|||
*INV:\twie war Ihr Wochenende ?
*PAR:\twas ?
*INV:\twie war Ihr Wochenende ?
*PAR:\t&-äh es war (..) ganz gut [/] gut .
*INV:\thaben Sie etwas unternommen ?
*PAR:\twir sind zum [//] in den Park gegangen .
@End
"""

out = Path(__file__).parent / "data"
(out / "dementia_eng").mkdir(parents=True, exist_ok=True)
(out / "dialogue_deu").mkdir(parents=True, exist_ok=True)
(out / "dementia_eng" / "sample1.cha").write_text(SAMPLE_DEMENTIA, encoding="utf-8")
(out / "dialogue_deu" / "sample2.cha").write_text(SAMPLE_DIALOGUE, encoding="utf-8")
print("Wrote synthetic samples to", out)
