"""Answer normalisation and matching (SQuAD-style)."""

from __future__ import annotations

import re
import string
import unicodedata

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_text(text: object) -> str:
    """Lowercase, strip accents, drop punctuation and articles, collapse whitespace."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower().translate(_PUNCT)
    return " ".join(_ARTICLES.sub(" ", text).split())


def extract_numbers(text: object) -> list[float]:
    """All numbers in the text; thousands separators are removed first."""
    cleaned = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", str(text))
    return [float(m) for m in _NUMBER.findall(cleaned)]


def numbers_close(a: float, b: float, rel: float = 0.01, abs_tol: float = 0.051) -> bool:
    """Within 1% or 0.05 absolute (so 18.44 matches 18.4 and 0.0 matches 0)."""
    return abs(a - b) <= max(abs_tol, rel * abs(b))


def answer_matches(prediction: str, answers: list[str], answer_type: str) -> bool:
    """Does the predicted answer text contain a gold answer?

    - "yesno": the first word of the prediction must be the gold yes/no.
    - "number": some number in the prediction is within tolerance of the gold number.
    - "text": a normalised gold alias appears in the normalised prediction.
    """
    if not prediction or not prediction.strip():
        return False
    if answer_type == "yesno":
        words = normalize_text(prediction).split()
        return bool(words) and words[0] in {normalize_text(a) for a in answers}
    if answer_type == "number":
        golds = [float(a) for a in answers]
        return any(numbers_close(p, g) for p in extract_numbers(prediction) for g in golds)
    pred = normalize_text(prediction)
    return any(_text_match(pred, normalize_text(a)) for a in answers)


def _text_match(pred: str, gold: str) -> bool:
    """Gold appears in the prediction, or the prediction is a substantial sub-phrase of the gold.

    The second case covers gold answers that are longer than what a person would
    say, e.g. gold 'Robert "Rob" Parissi' vs. predicted "Rob Parissi": the
    prediction must be a contiguous part of the gold and cover at least half of it.
    """
    if not pred or not gold:
        return False
    if f" {gold} " in f" {pred} ":
        return True
    return f" {pred} " in f" {gold} " and 2 * len(pred.split()) >= len(gold.split())
