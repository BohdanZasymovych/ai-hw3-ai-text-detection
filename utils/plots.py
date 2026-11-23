"""Shared plotting helpers with fixed label order and colors."""
import numpy as np
import pandas as pd

from .config import LABEL_COLORS, LABEL_ORDER


def plot_label_share(df: pd.DataFrame, by: str, ax) -> None:
    """100% stacked horizontal bars: label share (%) for each value of column `by`."""
    share = pd.crosstab(df[by], df["label"], normalize="index")[LABEL_ORDER] * 100
    share.plot.barh(stacked=True, color=LABEL_COLORS, ax=ax, width=0.7, legend=False)
    for container in ax.containers:
        ax.bar_label(container, labels=[f"{v:.0f}%" if v >= 4 else "" for v in container.datavalues],
                     label_type="center", fontsize=8, color="white")
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of texts, %")
    ax.set_ylabel("")
    ax.invert_yaxis()


def plot_compare(table: pd.DataFrame, ax, colors: dict) -> None:
    """Grouped horizontal bars of macro-F1 per group (rows) for each model (columns); `n` goes into the labels."""
    scores = table.drop(columns="n")
    names = ["/".join(k) if isinstance(k, tuple) else str(k) for k in table.index]
    scores.index = [f"{name} (n={n:,})" for name, n in zip(names, table["n"])]
    scores.plot.barh(ax=ax, color=[colors[c] for c in scores.columns], width=0.8)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=7)
    ax.set_xlim(0, 1.12)  # room for the value labels
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("macro-F1")
    ax.set_ylabel("")
    ax.invert_yaxis()
    n_models = len(scores.columns)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),  # above the plot
              ncol=n_models if n_models <= 3 else 2)
