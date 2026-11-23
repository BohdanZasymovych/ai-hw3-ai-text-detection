"""Stacker over model scores and features, scored out-of-fold on the validation set.

The base models never saw validation labels, so their validation scores are fair stacker inputs; the stacker itself
is fitted on 4/5 of the validation rows and predicts the remaining 1/5, so every row is predicted by a stacker that
did not see it. Two stacker kinds: "logreg" (logistic regression on standardised inputs) and "boosting"
(histogram gradient boosting).
"""
import itertools

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import macro_f1


def make_stacker(kind: str = "logreg", seed: int = 0):
    if kind == "boosting":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    raise ValueError(f"unknown stacker kind {kind}")


def oof_predict(X, y, seed: int, n_splits: int = 5, kind: str = "logreg") -> np.ndarray:
    """Out-of-fold predicted labels for the validation rows."""
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(make_stacker(kind, seed), np.nan_to_num(X), y, cv=cv)


def fit_predict(X, y, X_test, kind: str = "logreg", seed: int = 0) -> np.ndarray:
    """Stacker fitted on all validation rows, predicting the test rows."""
    return make_stacker(kind, seed).fit(np.nan_to_num(X), y).predict(np.nan_to_num(X_test))


def combinations(names) -> list:
    """All non-empty combinations of input group names, e.g. ("L",), ("L", "T"), ..."""
    return [c for r in range(1, len(names) + 1) for c in itertools.combinations(names, r)]


def choose(scores: dict, sizes: dict, tol: float):
    """The combination with the fewest inputs among those within `tol` of the best score (ties: higher score)."""
    best = max(scores.values())
    return min((c for c in scores if scores[c] >= best - tol), key=lambda c: (sizes[c], -scores[c]))


def nested_selection(groups: dict, y, seed: int, tol: float, n_splits: int = 5):
    """Honest score of "run the grid and choose": in each outer fold, the combination is chosen by out-of-fold
    scores on the other folds only, then a stacker on it predicts the held-out fold. Returns the predictions and
    the combination chosen in each outer fold."""
    combos = combinations(list(groups))
    sizes = {c: sum(groups[g].shape[1] for g in c) for c in combos}
    pred, chosen = np.empty(len(y), dtype=object), []
    for train, held_out in StratifiedKFold(n_splits, shuffle=True, random_state=seed + 1).split(y, y):
        stack = lambda c: np.hstack([groups[g] for g in c])
        scores = {c: macro_f1(y[train], oof_predict(stack(c)[train], y[train], seed)) for c in combos}
        best = choose(scores, sizes, tol)
        chosen.append(best)
        X = np.nan_to_num(stack(best))
        pred[held_out] = make_stacker("logreg", seed).fit(X[train], y[train]).predict(X[held_out])
    return pred, chosen


def window_features(logits: np.ndarray, doc: np.ndarray, n_windows: np.ndarray) -> np.ndarray:
    """Per text, from the logits of its windows (rows of `logits`, text index `doc`, windows sorted by position):
    the first, last, mean, min and max window logits (3 values each) and the log of the number of windows."""
    n = len(n_windows)
    starts = np.searchsorted(doc, np.arange(n))
    first, last = logits[starts], logits[starts + n_windows - 1]
    mean = np.zeros((n, logits.shape[1]))
    np.add.at(mean, doc, logits)
    mean /= n_windows[:, None]
    low, high = np.full((n, logits.shape[1]), np.inf), np.full((n, logits.shape[1]), -np.inf)
    np.minimum.at(low, doc, logits)
    np.maximum.at(high, doc, logits)
    return np.hstack([first, last, mean, low, high, np.log(n_windows)[:, None]])
