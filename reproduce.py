"""
Reproduce the two final submissions: python3 reproduce.py

Creates .venv and installs requirements (if not already in a venv), checks the data and the GPU,
runs the notebooks in order (cached steps are skipped) and checks that the submissions were saved.
Executed copies of the notebooks go to artifacts/executed/; the notebooks in the repo stay unchanged.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
NOTEBOOKS = ["03_encoder", "04_chunking", "06_large_model", "06b_large_windows", "07_final_pipeline"]
GPU_NOTEBOOKS = NOTEBOOKS[:4]  # 07 runs on CPU
SUBMISSIONS = ["submission_final.csv", "submission_final_logreg.csv"]


def step(text):
    print(f"\n=== {text}")


def run(*cmd, **kw):
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def setup_env():
    if sys.prefix != sys.base_prefix:
        return
    step("1. Environment: creating .venv and installing requirements")
    if not VENV.exists():
        run(sys.executable, "-m", "venv", VENV)
    python = VENV / "bin" / "python"
    run(python, "-m", "pip", "install", "-q", "-r", ROOT / "requirements.txt", "nbconvert")
    os.execv(python, [str(python), *sys.argv])  # continue inside the venv


def check_data():
    step("2. Data: train.csv, test.csv and sample_submission.csv must be in data/")
    sys.path.insert(0, str(ROOT))
    from utils.data import ensure_data
    from utils.paths import DATA_DIR
    try:
        ensure_data()  # downloads from Kaggle if the files are missing
    except Exception as e:
        sys.exit(f"Data not found in {DATA_DIR} and the Kaggle download failed ({e}).\n"
                 "Copy the three files there by hand, or put your Kaggle API token in ~/.kaggle/kaggle.json "
                 "and accept the competition rules.")
    print(f"OK: {DATA_DIR}")


def check_gpu():
    step(f"3. GPU: needed by {', '.join(GPU_NOTEBOOKS)} (training and inference of XLM-R)")
    import torch
    if not torch.cuda.is_available():
        sys.exit("No GPU found. Run this script on a machine with a CUDA GPU (e.g. Kaggle or Colab).")
    print("OK:", torch.cuda.get_device_name(0))


def run_notebooks():
    step("4. Notebooks (long: the two trainings take hours; saved results are reused if you rerun)")
    out = ROOT / "artifacts" / "executed"
    for name in NOTEBOOKS:
        print(f"-> {name}.ipynb", flush=True)
        run(sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--output-dir", out,
            "--ExecutePreprocessor.timeout=-1", f"{name}.ipynb", cwd=ROOT / "notebooks")


def check_results():
    step("5. Results")
    missing = [f for f in SUBMISSIONS if not (ROOT / "artifacts" / f).exists()]
    if missing:
        sys.exit(f"Missing in artifacts/: {', '.join(missing)}")
    print("Saved to artifacts/:", ", ".join(SUBMISSIONS))


if __name__ == "__main__":
    print(__doc__)
    setup_env()
    check_data()
    check_gpu()
    run_notebooks()
    check_results()
