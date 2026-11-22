"""Download the competition data if it is missing."""
import os
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


def submit_to_kaggle(path, message: str) -> dict:
    """Upload a submission file to the competition (counts toward the daily limit) and return what was sent.

    Locally uses ~/.kaggle credentials; on Kaggle, the KAGGLE_USERNAME and KAGGLE_KEY notebook secrets.
    """
    env = dict(os.environ)
    if ENV == "kaggle":
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        env.update({name: secrets.get_secret(name) for name in ("KAGGLE_USERNAME", "KAGGLE_KEY")})
    subprocess.run([sys.executable, "-m", "kaggle", "competitions", "submit", "-c", COMPETITION,
                    "-f", str(path), "-m", message], check=True, env=env)
    return {"file": str(path), "message": message}
