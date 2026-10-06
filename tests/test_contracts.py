from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax
from sklearn.dummy import DummyClassifier

import seed_classifier.predict as prediction
from seed_classifier.data import FEATURES, deduplicate, split_data
from seed_classifier.predict import predict, validate
from seed_classifier.support import range_diagnostics, training_feature_ranges
from seed_classifier.train import fit_temperature, scores


def fixture():
    # Artificial fixtures used only to exercise contracts.
    frame = pd.DataFrame(np.arange(40 * 16).reshape(40, 16) + 1, columns=FEATURES)
    frame["Class"] = ["a", "b"] * 20
    frame["source_row_id"] = range(40)
    return frame


def test_duplicate_copies_removed_before_split_and_conflicts_quarantined():
    f = fixture()
    unique, audit = deduplicate(pd.concat([f, f.iloc[:1]], ignore_index=True))
    assert len(unique) == 40 and audit["same_label_duplicate_rows_removed"] == 1
    conflict = f.iloc[:1].copy()
    conflict["Class"] = "b"
    unique, audit = deduplicate(pd.concat([f, conflict], ignore_index=True))
    assert len(unique) == 39 and audit["conflicting_label_rows_quarantined"] == 2


def test_stratified_split_has_disjoint_source_and_feature_ids():
    train, val, test = split_data(fixture())
    assert sum(map(len, [train, val, test])) == 40
    assert set(train.Class) == set(val.Class) == set(test.Class) == {"a", "b"}
    assert not set(train.source_row_id) & set(test.source_row_id)


def test_temperature_and_input_contract():
    logits = np.array([[3, 0], [0, 3], [2, 0], [0, 2]])
    t = fit_temperature(logits, ["a", "b", "a", "b"], ["a", "b"])
    assert 0.05 <= t <= 10 and np.allclose(softmax(logits / t, axis=1).sum(axis=1), 1)
    payload = {name: 1.0 for name in FEATURES}
    validate(payload)
    for bad in [{**payload, "extra": 1}, {**payload, "Area": True}, {**payload, "Area": np.nan}]:
        with pytest.raises(ValueError):
            validate(bad)


def test_actual_selected_artifact():
    import json

    root = Path(__file__).resolve().parents[1]
    if not (root / "models/selected.joblib").exists():
        pytest.skip("Actual trained model is not committed")
    result = predict(root, json.loads((root / "configs/example-input.json").read_text()))
    assert len(result["probabilities"]) == 7
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert result["variety"] in result["probabilities"]
    assert result["variety_probability"] == result["probabilities"][result["variety"]]


def test_training_ranges_exclude_holdout_outliers():
    train, validation, test = split_data(fixture())
    validation = validation.copy()
    test = test.copy()
    validation["Area"] = 1e10
    test["Area"] = 1e12
    ranges = training_feature_ranges(train)
    assert ranges["Area"] == {"min": float(train.Area.min()), "max": float(train.Area.max())}
    assert ranges["Area"]["max"] < min(validation.Area.min(), test.Area.min())


def test_support_boundaries_extrapolation_and_legacy_metadata():
    ranges = training_feature_ranges(fixture())
    for boundary in ["min", "max"]:
        payload = {name: ranges[name][boundary] for name in FEATURES}
        assert range_diagnostics(payload, ranges)["status"] == "within_training_ranges"
    payload = {name: ranges[name]["min"] for name in FEATURES}
    payload["Area"] = ranges["Area"]["max"] + 1
    result = range_diagnostics(payload, ranges)
    assert result["status"] == "outside_training_ranges"
    assert result["out_of_range_features"] == ["Area"]
    assert range_diagnostics(payload)["status"] == "unavailable"
    with pytest.raises(ValueError, match="metadata"):
        range_diagnostics(payload, {"Area": ranges["Area"]})


def test_support_diagnostics_do_not_change_predictions(monkeypatch, tmp_path):
    frame = fixture()
    model = DummyClassifier(strategy="prior").fit(frame[FEATURES], frame.Class)
    state = {"model": model, "classes": model.classes_.tolist(), "temperature": 1.5}
    ranges = training_feature_ranges(frame)
    for area in [frame.Area.iloc[0], ranges["Area"]["max"] * 2]:
        payload = frame[FEATURES].iloc[0].to_dict()
        payload["Area"] = float(area)
        monkeypatch.setattr(prediction.joblib, "load", lambda _path: state)
        legacy = predict(tmp_path, payload)
        assert legacy.pop("input_support")["status"] == "unavailable"
        enhanced = {**state, "training_feature_ranges": ranges}
        monkeypatch.setattr(prediction.joblib, "load", lambda _path, current=enhanced: current)
        result = predict(tmp_path, payload)
        support = result.pop("input_support")
        assert support["status"] == (
            "outside_training_ranges" if area > ranges["Area"]["max"] else "within_training_ranges"
        )
        assert result == legacy


def test_model_score_and_calibrated_probability_batch_invariance():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    frame = fixture()
    # Three classes exercise the multiclass score path, with fixed training statistics.
    labels = np.asarray(["a", "b", "c", "a"] * 10)
    model = make_pipeline(StandardScaler(), SVC(decision_function_shape="ovr"))
    model.fit(frame[FEATURES], labels)
    x = frame[FEATURES].iloc[[1, 5, 8]]
    alone = scores(model, x.iloc[[0]])
    batch = scores(model, x)[:1]
    np.testing.assert_allclose(alone, batch, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(softmax(alone / 0.7, axis=1), softmax(batch / 0.7, axis=1))


def test_legacy_upgrade_rejects_different_split_before_artifact_changes(tmp_path, monkeypatch):
    import json

    import scripts.add_training_support as upgrade_script

    frame = fixture()
    config = {"seed": 42}
    audit = {"source_rows": len(frame)}
    for name in ["configs", "reports", "models"]:
        (tmp_path / name).mkdir()
    (tmp_path / "configs/default.json").write_text(json.dumps(config))
    (tmp_path / "reports/metrics.json").write_text(
        json.dumps({"config": config, "source_audit": audit})
    )
    parts = split_data(frame)
    membership = pd.concat(
        [
            part[["source_row_id", "Class"]].assign(split=name)
            for name, part in zip(["train", "validation", "test"], parts, strict=True)
        ]
    ).reset_index(drop=True)
    membership.loc[0, "split"] = "test"  # Corrupt a recorded training row.
    membership.to_csv(tmp_path / "reports/split-membership.csv", index=False)
    artifact = tmp_path / "models/selected.joblib"
    artifact.write_bytes(b"Do not load or overwrite an unverified artifact")
    monkeypatch.setattr(upgrade_script, "read_source", lambda _root: (frame, audit))
    with pytest.raises(AssertionError):
        upgrade_script.upgrade(tmp_path)
    assert artifact.read_bytes() == b"Do not load or overwrite an unverified artifact"
    assert not (tmp_path / "models/selected.before-support.joblib").exists()
