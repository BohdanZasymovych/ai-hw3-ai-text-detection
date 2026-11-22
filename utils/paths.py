"""Environment detection and shared paths (local / Kaggle / Colab).

Override any location with env vars HW3_DATA_DIR and HW3_ARTIFACTS_DIR.
"""
import os
import sys
from pathlib import Path

REQUIRED_DATA_FILES = ("train.csv", "test.csv", "sample_submission.csv")


def detect_env() -> str:
    if Path("/kaggle").exists():
        return "kaggle"
    if "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules:
        return "colab"
    return "local"


ENV = detect_env()
REPO_ROOT = Path(__file__).resolve().parent.parent


def _find_kaggle_data() -> Path:
    """Competition data attached as a Kaggle input (path layout varies)."""
    for train in Path("/kaggle/input").rglob("train.csv"):
        return train.parent
    return Path("/kaggle/input")


def _default_artifacts() -> Path:
    if ENV == "kaggle":
        return Path("/kaggle/working/artifacts")
    if ENV == "colab":
        drive = Path("/content/drive/MyDrive/hw3")
        # Drive persists across sessions; fall back to the (ephemeral) local disk.
        return drive / "artifacts" if drive.parent.exists() else Path("/content/artifacts")
    return REPO_ROOT / "artifacts"


def _default_data() -> Path:
    if ENV == "kaggle":
        return _find_kaggle_data()
    if ENV == "colab":
        return Path("/content/data")
    return REPO_ROOT / "data"


DATA_DIR = Path(os.environ.get("HW3_DATA_DIR") or _default_data())
ARTIFACTS_DIR = Path(os.environ.get("HW3_ARTIFACTS_DIR") or _default_artifacts())

# Read-only fallbacks: artifacts from earlier Kaggle runs attached as datasets/inputs.
EXTRA_ARTIFACT_DIRS = (
    [p for p in Path("/kaggle/input").glob("*/artifacts") if p.is_dir()]
    + [p for p in Path("/kaggle/input").glob("*/*/artifacts") if p.is_dir()]
    if ENV == "kaggle"
    else []
)


def data_present() -> bool:
    return all((DATA_DIR / f).exists() for f in REQUIRED_DATA_FILES)


def ensure_dirs() -> None:
    """Create writable dirs. Never touches the read-only /kaggle/input."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if ENV != "kaggle":
        DATA_DIR.mkdir(parents=True, exist_ok=True)
