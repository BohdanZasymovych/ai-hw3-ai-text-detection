"""Run at the top of every notebook, before any third-party import (standard library only).

Packages whose cached pickles move between machines must have the same version everywhere.
On Kaggle/Colab, installs the versions pinned in requirements.txt; locally, only warns.
"""
import importlib.metadata as md
import os
import subprocess
import sys
from pathlib import Path

MUST_MATCH = {"scikit-learn": "sklearn"}  # pip name -> import name
REQUIREMENTS = Path(__file__).with_name("requirements.txt")
LOCAL = not (Path("/kaggle").exists() or "COLAB_RELEASE_TAG" in os.environ)


def _pins() -> dict:
    pairs = (line.split("==") for line in REQUIREMENTS.read_text().splitlines() if "==" in line)
    return {name.strip().lower(): version.strip() for name, version in pairs if name.strip().lower() in MUST_MATCH}


def _installed(name: str):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def install_pins() -> None:
    wrong = {name: version for name, version in _pins().items() if _installed(name) != version}
    if not wrong:
        return
    specs = [f"{name}=={version}" for name, version in wrong.items()]
    if LOCAL:
        print(f"[setup] version mismatch, run `pip install -r requirements.txt`: {specs}")
        return
    print(f"[setup] installing {specs}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *specs])
    if any(MUST_MATCH[name] in sys.modules for name in wrong):
        raise RuntimeError("[setup] packages were already imported: restart the kernel and rerun this cell")
