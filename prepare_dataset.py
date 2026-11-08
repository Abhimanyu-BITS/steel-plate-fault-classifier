"""
prepare_dataset.py
------------------
Step 1 of the pipeline: pull the raw *Steel Plates Faults* study from the UCI
Machine Learning Repository and reshape it into a single tidy CSV that the
training script can consume.

The archive ships two files:

    Faults.NNA        1941 tab-separated rows, no header, 34 columns
    Faults27x7_var    the 34 column names, one per line

Columns 1..27  -> geometric / photometric descriptors of a surface defect
Columns 28..34 -> a one-hot block naming the fault category

A one-hot target block is awkward for scikit-learn's single-label estimators,
so the seven indicator columns are collapsed back into one categorical
`Fault_Type` column.

Run:  python prepare_dataset.py
Out:  data/raw/...                 (untouched UCI files, kept for provenance)
      data/steel_faults_full.csv   (1941 x 28, ready for modelling)
"""

from __future__ import annotations

import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

UCI_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip"

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TIDY_CSV = PROJECT_ROOT / "data" / "steel_faults_full.csv"

# The seven mutually exclusive surface-defect categories, in the column order
# used by the original Semeion study.
FAULT_CATEGORIES = [
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
]

TARGET_COLUMN = "Fault_Type"


def download_archive(destination: Path) -> None:
    """Fetch and unpack the UCI zip into `destination` (skipped if present)."""
    destination.mkdir(parents=True, exist_ok=True)
    expected = [destination / "Faults.NNA", destination / "Faults27x7_var"]
    if all(f.exists() for f in expected):
        print(f"[prepare] raw files already in {destination}, skipping download")
        return

    print(f"[prepare] downloading {UCI_ARCHIVE_URL}")
    with urllib.request.urlopen(UCI_ARCHIVE_URL, timeout=120) as response:
        payload = response.read()
    print(f"[prepare] received {len(payload):,} bytes")

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(destination)
    print(f"[prepare] extracted -> {[p.name for p in destination.iterdir()]}")


def load_raw_frame(source: Path) -> pd.DataFrame:
    """Read Faults.NNA and attach the column names from Faults27x7_var."""
    header_file = source / "Faults27x7_var"
    data_file = source / "Faults.NNA"

    column_names = [
        line.strip()
        for line in header_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(column_names) != 34:
        raise ValueError(f"expected 34 column names, found {len(column_names)}")

    frame = pd.read_csv(data_file, sep="\t", header=None, names=column_names)
    print(f"[prepare] raw frame shape = {frame.shape}")
    return frame


def collapse_one_hot_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn the 7 indicator columns into a single categorical target column."""
    indicator_block = frame[FAULT_CATEGORIES]

    # Sanity check: every row must belong to exactly one category.
    row_sums = indicator_block.sum(axis=1)
    if not (row_sums == 1).all():
        offenders = int((row_sums != 1).sum())
        raise ValueError(f"{offenders} row(s) are not single-labelled")

    feature_block = frame.drop(columns=FAULT_CATEGORIES)
    labels = indicator_block.idxmax(axis=1)

    tidy = feature_block.copy()
    tidy[TARGET_COLUMN] = labels
    return tidy


def main() -> int:
    download_archive(RAW_DIR)
    raw = load_raw_frame(RAW_DIR)
    tidy = collapse_one_hot_target(raw)

    TIDY_CSV.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(TIDY_CSV, index=False)

    print(f"\n[prepare] tidy dataset written to {TIDY_CSV.relative_to(PROJECT_ROOT)}")
    print(f"[prepare] instances = {len(tidy):,}  features = {tidy.shape[1] - 1}")
    print("[prepare] class balance:")
    counts = tidy[TARGET_COLUMN].value_counts()
    for name, n in counts.items():
        print(f"           {name:<14} {n:>5}  ({n / len(tidy):6.2%})")

    # Assignment floor: >= 12 features and >= 500 instances.
    assert tidy.shape[1] - 1 >= 12, "feature count below assignment minimum"
    assert len(tidy) >= 500, "instance count below assignment minimum"
    print("\n[prepare] assignment minimums satisfied (>=12 features, >=500 instances)")

    summary = {
        "source_url": UCI_ARCHIVE_URL,
        "n_instances": int(len(tidy)),
        "n_features": int(tidy.shape[1] - 1),
        "target_column": TARGET_COLUMN,
        "classes": {str(k): int(v) for k, v in counts.items()},
        "feature_names": [c for c in tidy.columns if c != TARGET_COLUMN],
    }
    (PROJECT_ROOT / "data" / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
