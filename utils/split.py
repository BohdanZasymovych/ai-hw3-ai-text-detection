"""Fixed train/validation split, made once and cached, shared by all notebooks."""
import numpy as np
import pandas as pd

from .cache import cached
from .paths import DATA_DIR

STRATA = ["lang", "domain", "label"]


def get_split(val_frac: float = 0.1, seed: int = 42) -> dict:
    """Row positions in train.csv: {"train": ..., "val": ...}.

    Stratified by language, domain and label, so every cell is represented in both parts.
    Loaded from the cache if it exists, otherwise made and saved. Use the positions with
    `train.iloc[split["train"]]`.
    """
    def make():
        meta = pd.read_csv(DATA_DIR / "train.csv", usecols=STRATA)
        val = meta.groupby(STRATA, group_keys=False).sample(frac=val_frac, random_state=seed).index
        train = meta.index.difference(val)
        return {"train": train.to_numpy(), "val": np.sort(val.to_numpy())}

    key = {"val_frac": val_frac, "seed": seed, "strata": STRATA, "v": 1}
    return cached("split", make, key=key)
