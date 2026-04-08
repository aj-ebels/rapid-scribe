"""
Text normalization and simple accuracy metrics for transcription evaluation (WER / CER).
"""
from __future__ import annotations

import re
import unicodedata


def normalize_for_compare(text: str) -> str:
    """Lowercase, NFKC, strip surrounding whitespace, collapse internal spaces."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = t.lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _levenshtein(a: list, b: list) -> int:
    """Classic DP edit distance on sequences of hashable items."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = cur
    return prev[m]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Word error rate = (S+D+I) / max(1, len(reference words)).
    Uses whitespace tokenization after normalize_for_compare.
    """
    ref = normalize_for_compare(reference).split()
    hyp = normalize_for_compare(hypothesis).split()
    if not ref:
        return 1.0 if hyp else 0.0
    dist = _levenshtein(ref, hyp)
    return dist / len(ref)


def char_error_rate(reference: str, hypothesis: str) -> float:
    """Character error rate on normalized strings (no spaces collapsed to single — normalize does)."""
    ref = normalize_for_compare(reference)
    hyp = normalize_for_compare(hypothesis)
    ra, ha = list(ref), list(hyp)
    if not ra:
        return 1.0 if ha else 0.0
    return _levenshtein(ra, ha) / len(ra)
