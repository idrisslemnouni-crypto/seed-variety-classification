"""Upgrade a trusted legacy artifact after checking its recorded source/splits.

Run from the repository root with PYTHONPATH=src. No estimator is fitted.
"""

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from seed_classifier.data import FEATURES, read_source, split_data
from seed_classifier.predict import predict
from seed_classifier.support import training_feature_ranges


def upgrade(root):
    config = json.loads((root / "configs/default.json").read_text())
    metrics = json.loads((root / "reports/metrics.json").read_text())
    if config != metrics["config"]:
        raise ValueError("Experiment config differs from the recorded run")
    frame, audit = read_source(root)  # Verifies the official source checksum.
    if audit != metrics["source_audit"]:
        raise ValueError("Source audit differs from the recorded run")
    partitions = dict(
        zip(["train", "validation", "test"], split_data(frame, config["seed"]), strict=True)
    )
    expected = (
        pd.concat(
            [
                part[["source_row_id", "Class"]].assign(split=name)
                for name, part in partitions.items()
            ]
        )
        .sort_values("source_row_id")
        .reset_index(drop=True)
    )
    recorded = (
        pd.read_csv(root / "reports/split-membership.csv")
        .sort_values("source_row_id")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(expected, recorded, check_dtype=False)
    artifact = root / "models/selected.joblib"
    state = joblib.load(artifact)
    if (
        state["features"] != FEATURES
        or state["classes"] != state["model"].classes_.tolist()
        or state["selected"] != metrics["selected_on_validation"]
        or state["temperature"] != metrics["temperature_fit_on_validation"]
    ):
        raise ValueError("Artifact metadata differs from the recorded run")
    if state["selected"] == "rbf_svm":
        # Check retained training vectors against this exact ordered training split.
        pipeline = state["model"]
        svc = pipeline.named_steps["svc"]
        scaled = pipeline[:-1].transform(partitions["train"][FEATURES])
        np.testing.assert_allclose(svc.support_vectors_, scaled[svc.support_], rtol=0, atol=1e-12)
    ranges = training_feature_ranges(partitions["train"])
    if "training_feature_ranges" in state:
        if state["training_feature_ranges"] != ranges:
            raise ValueError("Existing support metadata differs from the verified training split")
        return {"status": "already_present", "training_rows": len(partitions["train"])}
    payload = json.loads((root / "configs/example-input.json").read_text())
    before = predict(root, payload)
    before.pop("input_support")
    backup = artifact.with_name("selected.before-support.joblib")
    if backup.exists():
        raise ValueError("Legacy artifact backup already exists; preserve it before upgrading")
    shutil.copy2(artifact, backup)
    temporary = artifact.with_suffix(".support.part")
    state["training_feature_ranges"] = ranges
    try:
        joblib.dump(state, temporary)
        temporary.replace(artifact)
        after = predict(root, payload)
        support = after.pop("input_support")
        if after != before:
            raise ValueError("Upgrade changed the example classification or probabilities")
    except Exception:
        shutil.copy2(backup, artifact)
        temporary.unlink(missing_ok=True)
        raise
    return {
        "status": "upgraded_without_refit",
        "source_rows": audit["source_rows"],
        "verified_split_rows": {name: len(part) for name, part in partitions.items()},
        "training_features": len(ranges),
        "example_prediction_and_probabilities_unchanged": True,
        "example_input_support": support,
        "backup": backup.relative_to(root).as_posix(),
    }


if __name__ == "__main__":
    print(json.dumps(upgrade(Path.cwd()), indent=2))
