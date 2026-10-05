"""Hash-pinned official ARFF and duplicate-aware grain preparation."""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split

FEATURES = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]


def deduplicate(frame):
    counts = frame.groupby(FEATURES, dropna=False).Class.transform("nunique")
    conflicts = frame[counts > 1]
    eligible = frame[counts == 1]
    unique = eligible.drop_duplicates(FEATURES, keep="first").copy()
    audit = {
        "source_rows": len(frame),
        "same_label_duplicate_rows_removed": len(eligible) - len(unique),
        "conflicting_label_rows_quarantined": len(conflicts),
        "retained_unique_rows": len(unique),
        "missing_cells": int(frame[FEATURES + ["Class"]].isna().sum().sum()),
    }
    return unique, audit


def read_source(root: Path):
    manifest = json.loads((root / "data/source-manifest.json").read_text())
    raw = root / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "dry-bean.zip"
    if not archive.exists():
        with urllib.request.urlopen(manifest["url"], timeout=60) as response:
            content = response.read()
        if (
            len(content) != manifest["bytes"]
            or hashlib.sha256(content).hexdigest() != manifest["sha256"]
        ):
            raise ValueError("Official source archive changed")
        archive.write_bytes(content)
    if (
        archive.stat().st_size != manifest["bytes"]
        or hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["sha256"]
    ):
        raise ValueError("Source checksum mismatch")
    with zipfile.ZipFile(archive) as source:
        for name in ["Dry_Bean_Dataset.arff", "Dry_Bean_Dataset.txt"]:
            (raw / name).write_bytes(source.read("DryBeanDataset/" + name))
    values, _ = arff.loadarff(raw / "Dry_Bean_Dataset.arff")
    frame = pd.DataFrame(values)
    frame["Class"] = frame.Class.str.decode("ascii")
    frame["source_row_id"] = np.arange(len(frame))
    if (
        list(frame.columns[:16]) != FEATURES
        or not np.isfinite(frame[FEATURES]).all().all()
        or (frame[FEATURES] <= 0).any().any()
    ):
        raise ValueError("Unexpected morphology schema or numeric range")
    return deduplicate(frame)


def split_data(frame, seed=42):
    train, remaining = train_test_split(
        frame, test_size=0.3, random_state=seed, stratify=frame.Class
    )
    validation, test = train_test_split(
        remaining, test_size=0.5, random_state=seed, stratify=remaining.Class
    )
    for a, b in [(train, validation), (train, test), (validation, test)]:
        if set(a.source_row_id) & set(b.source_row_id) or set(
            pd.util.hash_pandas_object(a[FEATURES], index=False)
        ) & set(pd.util.hash_pandas_object(b[FEATURES], index=False)):
            raise ValueError("Split identity or exact-feature overlap")
    return train, validation, test
