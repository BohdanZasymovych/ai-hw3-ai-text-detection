"""Download the competition data if it is missing."""
import subprocess
import sys
import zipfile

from .paths import DATA_DIR, ENV, data_present

COMPETITION = "ai-generated-text-detection-hw3-ucu-ai-2026"


def ensure_data() -> None:
    """Download and unpack the data into DATA_DIR unless it is already there."""
    if data_present():
        return
    if ENV == "kaggle":
        raise FileNotFoundError("Attach the competition data as a notebook input (Add Input).")
    # Needs Kaggle credentials and accepted competition rules.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(DATA_DIR)],
        check=True,
    )
    zip_path = DATA_DIR / f"{COMPETITION}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_DIR)
        zip_path.unlink()
    print("Downloaded:", sorted(p.name for p in DATA_DIR.iterdir()))
