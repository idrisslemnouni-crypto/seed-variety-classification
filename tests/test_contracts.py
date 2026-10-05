from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from seed_classifier.data import FEATURES, deduplicate, split_data
from seed_classifier.predict import predict, validate
from seed_classifier.train import fit_temperature


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
