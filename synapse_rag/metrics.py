"""Answer metrics (SQuAD-style EM / token F1), as in the QASA research evaluation."""

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not truth_tokens:
        return float(pred_tokens == truth_tokens)
    common = sum((Counter(pred_tokens) & Counter(truth_tokens)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(pred_tokens), common / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)
