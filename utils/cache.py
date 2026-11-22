"""Cache results of long steps (training, feature extraction, inference) on disk.

    KEY = {"model": "xlmr-base", "n": 30000, "seed": 42, "max_len": 512}
    model = cached("tfidf_svm", lambda: fit_svm(X, y), key={"C": 1.0, "n": 30000})
    logits = cached("xlmr_val_logits", lambda: predict(val), key=KEY, kind="npy")
    cached("xlmr_val_metrics", lambda: {...}, key=KEY, kind="json")

`key` lists everything that changes the result (hyperparameters, sample size,
seed, preprocessing version). It is saved next to the result as <name>.key.json;
if the saved key differs from the current one, the step is recomputed and overwritten.
Code or data changes that are not in `key` are NOT detected: add them to the key
(e.g. a "v": 2 entry you bump). Pass force=True or delete the file to recompute anyway.
Transformers: path = artifact_dir(name, key=KEY), then save_pretrained / from_pretrained.
"""
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .paths import ARTIFACTS_DIR, EXTRA_ARTIFACT_DIRS, ensure_dirs

_EXT = {"pickle": ".pkl", "npy": ".npy", "json": ".json"}


def _save(obj: Any, path: Path, kind: str) -> None:
    tmp = path.with_name(path.name + ".tmp")  # write then rename: no half-written files
    if kind == "npy":
        with open(tmp, "wb") as f:
            np.save(f, obj)
    elif kind == "json":
        tmp.write_text(json.dumps(obj, indent=2))
    else:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _load(path: Path, kind: str) -> Any:
    if kind == "npy":
        return np.load(path)
    if kind == "json":
        return json.loads(path.read_text())
    with open(path, "rb") as f:
        return pickle.load(f)


def _key_path(base: Path, name: str) -> Path:
    return base / f"{name}.key.json"


def _norm_key(key: dict) -> dict:
    return json.loads(json.dumps(key, sort_keys=True, default=str))  # canonical, comparable form


def _key_matches(base: Path, name: str, key: dict) -> bool:
    path = _key_path(base, name)
    return path.exists() and json.loads(path.read_text()) == _norm_key(key)


def _write_key(name: str, key: dict) -> None:
    _key_path(ARTIFACTS_DIR, name).write_text(json.dumps(_norm_key(key), indent=2))


def artifact_path(name: str, kind: str = "pickle") -> Path:
    return ARTIFACTS_DIR / f"{name}{_EXT[kind]}"


def find_artifact(name: str, kind: str = "pickle") -> Path | None:
    """Path of a result saved by another notebook (locally or in an attached Kaggle input), or None."""
    for base in (ARTIFACTS_DIR, *EXTRA_ARTIFACT_DIRS):
        path = base / f"{name}{_EXT[kind]}"
        if path.exists():
            return path
    return None


def artifact_dir(name: str, key: dict) -> Path:
    """Directory for things with their own format (HF checkpoints). Reuses an existing
    copy (also from attached Kaggle inputs) only if its saved key matches `key`;
    a stale local copy is emptied. Skip training if the returned dir is non-empty."""
    for base in (ARTIFACTS_DIR, *EXTRA_ARTIFACT_DIRS):
        path = base / name
        if path.is_dir() and any(path.iterdir()) and _key_matches(base, name, key):
            return path
    path = ARTIFACTS_DIR / name
    if path.exists():
        print(f"[cache] key changed, clearing {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    _write_key(name, key)
    return path


def cached(name: str, fn: Callable[[], Any], *, key: dict, kind: str = "pickle", force: bool = False) -> Any:
    """Return the saved result for `name` if its saved key matches `key`; otherwise
    compute it with `fn()`, save it (and the key) and return it."""
    if kind not in _EXT:
        raise ValueError(f"kind must be one of {list(_EXT)}")
    ensure_dirs()
    if not force:
        for base in (ARTIFACTS_DIR, *EXTRA_ARTIFACT_DIRS):
            path = base / f"{name}{_EXT[kind]}"
            if path.exists() and _key_matches(base, name, key):
                print(f"[cache] loaded {path}")
                return _load(path, kind)
            if path.exists():
                print(f"[cache] key changed or missing for {path}, recomputing")
    result = fn()
    path = artifact_path(name, kind)
    _save(result, path, kind)
    _write_key(name, key)
    print(f"[cache] saved {path}")
    return result
