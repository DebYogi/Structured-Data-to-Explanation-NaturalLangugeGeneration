"""Utility helpers for SAR output validation and normalization.
"""
from typing import Any
from .serializer import example_target_text


def detect_serialized_output(text: str) -> bool:
    """Return True if text looks like a serialized/tokenized output rather than human-readable SAR.

    Heuristics used:
    - Presence of serialized markers like 'TXN_', 'AGGREGATES:', 'ALERT=', 'REASON=', or 'TOTAL_DEBIT='
    - Presence of many '=' signs or pipe '|' separators
    - Very few sentence-ending punctuation marks ('.') compared to token length
    - Absence of the header 'SUSPICIOUS ACTIVITY REPORT'
    """
    if not text or len(text.strip()) < 20:
        return True
    markers = ["TXN_", "AGGREGATES:", "ALERT=", "REASON=", "TOTAL_DEBIT="]
    if any(m in text for m in markers):
        return True
    # If the text uses pipe separators a lot, it's likely serialized
    if text.count("|") >= 2:
        return True
    # If it contains many '=' signs, likely key=value serialization
    if text.count("=") > 3:
        return True
    # Check for explicit human-readable header
    if "SUSPICIOUS ACTIVITY REPORT" in text.upper():
        return False
    # If there are very few sentence endings for the amount of text, assume serialized
    sent_count = text.count('.')
    words = len(text.split())
    if words > 50 and sent_count < max(1, words // 50):
        return True
    return False


def ensure_human_readable_sar(out: str, account_id: str, txs: Any, agg: Any) -> str:
    """Return a human-readable SAR: keep `out` if it looks human, otherwise use deterministic template.
    """
    if detect_serialized_output(out):
        return example_target_text(account_id, txs, agg)
    return out
