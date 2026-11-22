"""Shared plotting helpers with fixed label order and colors."""
import pandas as pd

LABEL_ORDER = ["human", "ai", "fusion"]
LABEL_COLORS = {"human": "#4C78A8", "ai": "#E45756", "fusion": "#F2A93B"}


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
