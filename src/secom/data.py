"""Download and load the UCI SECOM dataset."""
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/"
FILES = ("secom.data", "secom_labels.data", "secom.names")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"


def download(raw_dir: Path = RAW_DIR, force: bool = False) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = raw_dir / name
        if force or not target.exists():
            print(f"Downloading {name} ...")
            urlretrieve(BASE_URL + name, target)


def load(raw_dir: Path = RAW_DIR):
    """Return (X, y, timestamps).

    X: 1567 x 590 sensor readings, columns named sensor_000..sensor_589
       (zero-based position in secom.data; the dataset ships no real names).
    y: 1 = wafer failed in-house line testing, 0 = passed (raw file uses 1 / -1).
    """
    download(raw_dir)
    X = pd.read_csv(raw_dir / "secom.data", sep=" ", header=None, na_values="NaN")
    X.columns = [f"sensor_{i:03d}" for i in range(X.shape[1])]

    labels = pd.read_csv(
        raw_dir / "secom_labels.data", sep=" ", header=None, names=["label", "timestamp"]
    )
    y = (labels["label"] == 1).astype(int).rename("fail")
    timestamps = pd.to_datetime(labels["timestamp"], format="%d/%m/%Y %H:%M:%S")
    return X, y, timestamps


if __name__ == "__main__":
    X, y, ts = load()
    print(f"X: {X.shape}, failures: {y.sum()} / {len(y)}, {ts.min()} -> {ts.max()}")
