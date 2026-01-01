"""Evaluation utilities: BLEU/ROUGE plus rule-based validation for hallucinations and traceability."""
from typing import List
from rouge_score import rouge_scorer
import sacrebleu


def compute_rouge(reference: str, hypothesis: str):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
    return scorer.score(reference, hypothesis)


def compute_bleu(refs: List[str], hyps: List[str]):
    return sacrebleu.corpus_bleu(hyps, [refs])


def rule_based_traceability_check(input_text: str, output_text: str) -> dict:
    # Conservative checks: ensure amounts present in input appear in output (string match), dates and CP IDs
    issues = []
    # Extract amounts (simple regex) -- more sophisticated parsing can be added
    import re
    AMT = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b")
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", input_text)
    amounts = AMT.findall(input_text)
    cps = re.findall(r"CP_\d+", input_text)
    for d in dates:
        if d not in output_text:
            issues.append(f"date_missing:{d}")
    for a in amounts:
        if a not in output_text:
            issues.append(f"amount_missing:{a}")
    for c in cps:
        if c not in output_text:
            issues.append(f"cp_missing:{c}")
    return {"ok": len(issues) == 0, "issues": issues}
