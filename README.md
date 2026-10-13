# SECOM wafer failure prediction

LightGBM classifier on the [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom) dataset:
1,567 wafers, 590 anonymous sensor signals, 104 failures (6.6%).

## Setup

macOS needs the OpenMP runtime for LightGBM: `brew install libomp`.

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Run

```bash
.venv/bin/python -m secom.explore            # missing values, distributions, imbalance -> reports/*.png
.venv/bin/python -m secom.train              # both imbalance strategies, stratified random split
.venv/bin/python -m secom.train --split time # train on earliest 80%, test on latest 20%
.venv/bin/python -m secom.train --beta 2     # pick a threshold that favours recall
```

The data is downloaded to `data/raw/` on first use.

## Layout

- `src/secom/data.py` – download + load; sensors are named `sensor_000`..`sensor_589` by column position
- `src/secom/explore.py` – exploratory summary and figures
- `src/secom/pipeline.py` – column filter (>50% missing or constant) → median imputation →
  (scaling + SMOTE) or class weighting → LightGBM
- `src/secom/train.py` – CV, threshold selection, hold-out precision / recall / ROC-AUC / PR-AUC, top-10 sensors
