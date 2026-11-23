"""Logistic-regression stacker over model scores and features, scored out-of-fold on the validation set.

The base models never saw validation labels, so their validation scores are fair stacker inputs; the stacker itself
is fitted on 4/5 of the validation rows and predicts the remaining 1/5, so every row is predicted by a stacker that
did not see it.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def make_stacker(C: float = 1.0) -> Pipeline:
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def oof_predict(X, y, seed: int, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold predicted labels for the validation rows."""
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(make_stacker(), np.nan_to_num(X), y, cv=cv)


def fit_predict(X, y, X_test) -> np.ndarray:
    """Stacker fitted on all validation rows, predicting the test rows."""
    return make_stacker().fit(np.nan_to_num(X), y).predict(np.nan_to_num(X_test))


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
