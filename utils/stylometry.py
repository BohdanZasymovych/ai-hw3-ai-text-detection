"""Simple stylometric features and word statistics for comparing text classes."""
import re
from collections import Counter

import numpy as np

WORD = re.compile(r"\w+")
SENTENCE_END = re.compile(r"[.!?]+")
PUNCT = re.compile(r"[^\w\s]")
DIGIT = re.compile(r"\d")
UPPER = re.compile(r"[A-ZÄÖÜ]")


def _spread(values) -> float:
    """Relative variation (std / mean) of a feature across text chunks."""
    values = np.asarray(values, dtype=float)
    return float(values.std() / values.mean()) if values.mean() > 0 else np.nan


def style_features(text: str) -> dict:
    """Style statistics of one text. The two `* variation` features measure how much the
    style changes along the text (4 equal chunks): higher means a less uniform text."""
    words = WORD.findall(text)
    n, chars = max(len(words), 1), max(len(text), 1)
    first = [w.lower() for w in words[:100]]
    long_enough = len(words) >= 8
    word_len = [np.mean([len(w) for w in c]) for c in np.array_split(np.array(words, dtype=object), 4)] if long_enough else []
    size = len(text) // 4 or 1
    punct = [len(PUNCT.findall(text[i * size:(i + 1) * size])) / size for i in range(4)]
    return {
        "Sentence length": n / max(len(SENTENCE_END.findall(text)), 1),
        "Word length": sum(map(len, words)) / n,
        "Type-token ratio": len(set(first)) / max(len(first), 1),
        "Punctuation": len(PUNCT.findall(text)) / chars,
        "Digits": len(DIGIT.findall(text)) / chars,
        "Capital letters": len(UPPER.findall(text)) / chars,
        "Word length variation": _spread(word_len) if long_enough else np.nan,
        "Punctuation variation": _spread(punct),
    }


def top_words(texts_a, texts_b, k: int = 12, min_df: int = 50):
    """Words most typical of group A vs group B (smoothed log-odds of document frequency).
    Returns (top for A, top for B)."""
    def doc_freq(texts):
        return Counter(w for t in texts for w in set(WORD.findall(t.lower())))

    ca, cb = doc_freq(texts_a), doc_freq(texts_b)
    na, nb = len(texts_a), len(texts_b)
    score = {w: np.log((ca[w] + 1) / (na + 2)) - np.log((cb[w] + 1) / (nb + 2))
             for w in ca.keys() | cb.keys() if ca[w] + cb[w] >= min_df}
    ranked = sorted(score, key=score.get)
    return ranked[::-1][:k], ranked[:k]
