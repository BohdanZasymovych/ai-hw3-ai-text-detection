"""Shared evaluation helpers: per-class report, confusion matrix, macro-F1 per group, submission."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .paths import ARTIFACTS_DIR, ensure_dirs
from .config import LABEL_ORDER


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, labels=LABEL_ORDER, average="macro")


def class_report(y_true, y_pred) -> pd.DataFrame:
    """Precision, recall, F1 and support per label, plus a macro-average row."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABEL_ORDER, zero_division=0)
    report = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=LABEL_ORDER)
    report.loc["macro avg"] = [p.mean(), r.mean(), f.mean(), s.sum()]
    return report


def plot_confusion(y_true, y_pred, ax) -> None:
    """Confusion matrix: row-normalised colours, with the share and the count in each cell."""
    counts = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    share = counts / counts.sum(axis=1, keepdims=True)
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(LABEL_ORDER)):
        for j in range(len(LABEL_ORDER)):
            ax.text(j, i, f"{share[i, j]:.0%}\n({counts[i, j]:,})", ha="center", va="center",
                    color="white" if share[i, j] > 0.5 else "black")
    ticks = range(len(LABEL_ORDER))
    ax.set_xticks(ticks, LABEL_ORDER)
    ax.set_yticks(ticks, LABEL_ORDER)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")


def group_f1(meta: pd.DataFrame, y_pred, by) -> pd.DataFrame:
    """Macro-F1 and number of texts for each group of `meta` (needs a `label` column)."""
    df = meta.assign(pred=np.asarray(y_pred))
    rows = {key: {"macro_f1": macro_f1(g["label"], g["pred"]), "n": len(g)} for key, g in df.groupby(by, observed=True)}
    return pd.DataFrame.from_dict(rows, orient="index")


def plot_group_f1(table: pd.DataFrame, overall: float, ax) -> None:
    """Horizontal bars of macro-F1 per group (n texts in the label), with the overall score as a line."""
    names = ["/".join(k) if isinstance(k, tuple) else str(k) for k in table.index]
    labels = [f"{name} (n={n:,})" for name, n in zip(names, table["n"])]
    ax.barh(labels, table["macro_f1"], color="#4C78A8")
    ax.axvline(overall, color="black", ls="--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel("macro-F1")
    ax.invert_yaxis()


def write_submission(ids, labels, sample_submission: pd.DataFrame, name: str):
    """Save `id,label` in the order of sample_submission.csv and return the file path."""
    submission = pd.DataFrame({"id": np.asarray(ids), "label": np.asarray(labels)})
    assert submission["id"].is_unique and set(submission["id"]) == set(sample_submission["id"]), "ids differ"
    assert submission["label"].isin(LABEL_ORDER).all(), "unknown label"
    submission = submission.set_index("id").loc[sample_submission["id"]].reset_index()
    ensure_dirs()
    path = ARTIFACTS_DIR / f"submission_{name}.csv"
    submission.to_csv(path, index=False)
    return path


LENGTH_BANDS = {"≤510": 510, "510–1k": 1000, "1k–2k": 2000, ">2k": np.inf}  # encoder tokens per text


def length_band(n_tokens) -> pd.Categorical:
    """Length band of each text from its token count; the first band fits the encoder without truncation."""
    return pd.cut(np.asarray(n_tokens), [0, *LENGTH_BANDS.values()], labels=list(LENGTH_BANDS))


def compare_f1(y_true, preds: dict) -> pd.DataFrame:
    """F1 per class and macro-F1 (rows) for each model's predictions (columns)."""
    return pd.DataFrame({name: class_report(y_true, pred)["f1"] for name, pred in preds.items()})


def compare_groups(meta: pd.DataFrame, preds: dict, by) -> pd.DataFrame:
    """Macro-F1 per group of `meta` (needs a `label` column) for each model, plus the number of texts `n`."""
    table = pd.DataFrame({name: group_f1(meta, pred, by)["macro_f1"] for name, pred in preds.items()})
    table["n"] = meta.groupby(by, observed=True).size()
    return table
