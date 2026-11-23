"""Fixed train/validation split and training subsample, made once and cached, shared by all notebooks."""
import numpy as np
import pandas as pd

from .cache import cached
from .config import SEED, SUBSAMPLE_FLOOR, SUBSAMPLE_SIZE, VAL_FRAC
from .paths import DATA_DIR

STRATA = ["lang", "domain", "label"]


def _read_meta() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "train.csv", usecols=STRATA)


def get_split(val_frac: float = VAL_FRAC, seed: int = SEED) -> dict:
    """Row positions in train.csv: {"train": ..., "val": ...}.

    Stratified by language, domain and label, so every cell is represented in both parts.
    Loaded from the cache if it exists, otherwise made and saved. Use the positions with
    `train.iloc[split["train"]]`.
    """
    def make():
        meta = _read_meta()
        val = meta.groupby(STRATA, group_keys=False).sample(frac=val_frac, random_state=seed).index
        train = meta.index.difference(val)
        return {"train": train.to_numpy(), "val": np.sort(val.to_numpy())}

    key = {"val_frac": val_frac, "seed": seed, "strata": STRATA, "v": 1}
    return cached("split", make, key=key)


def get_subsample(n: int = SUBSAMPLE_SIZE, floor: int = SUBSAMPLE_FLOOR, seed: int = SEED,
                  name: str = "subsample") -> np.ndarray:
    """Row positions in train.csv of a fixed training subsample, taken from the train part only.

    Cells (language, domain, label) with fewer than `floor` train texts are kept in full;
    the rest of the `n` texts is a random sample of the other cells. Give another `n` its own `name`,
    so the fixed comparison subsample is not overwritten.
    """
    split = get_split()

    def make():
        meta = _read_meta().iloc[split["train"]]
        size = meta.groupby(STRATA)["label"].transform("size")
        rare = meta[size < floor]
        rest = meta[size >= floor].sample(n - len(rare), random_state=seed)
        return np.sort(pd.concat([rare, rest]).index.to_numpy())

    key = {"n": n, "floor": floor, "seed": seed, "n_val": len(split["val"]), "v": 1}
    return cached(name, make, key=key, kind="npy")
