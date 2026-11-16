import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("FD_DATA_DIR", ROOT / "data"))
OUT_DIR = Path(os.environ.get("FD_OUT_DIR", ROOT / "outputs"))

TEP_DIR = DATA_DIR / "TEP"
CWRU_DIR = DATA_DIR / "CWRU"
CWRU_LOADS_DIR = DATA_DIR / "CWRU_loads"


def out(*parts):
    p = OUT_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
