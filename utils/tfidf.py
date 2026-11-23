"""TF-IDF vectorizers of the baseline (word 1-2-grams and character 3-5-grams)."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def make_vectorizer(kind: str) -> TfidfVectorizer:
    if kind == "word":
        return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True, dtype=np.float32)
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=200_000,
                           sublinear_tf=True, lowercase=False, dtype=np.float32)
