"""Inference from trusted locally trained numeric-feature model."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax

from seed_classifier.data import FEATURES
from seed_classifier.train import scores


def validate(payload):
    if set(payload) != set(FEATURES):
        raise ValueError("Exact 16-feature source schema required")
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0
        for v in payload.values()
    ):
        raise ValueError("Finite positive source morphology values required")


def predict(root, payload):
    validate(payload)
    state = joblib.load(root / "models/selected.joblib")  # Trusted locally trained artifact only.
    x = pd.DataFrame([payload], columns=FEATURES)
    probability = softmax(scores(state["model"], x) / state["temperature"], axis=1)[0]
    variety = str(state["model"].predict(x)[0])
    return {
        "variety": variety,
        "variety_probability": float(probability[state["classes"].index(variety)]),
        "probability_argmax_variety": state["classes"][int(probability.argmax())],
        "probabilities": dict(zip(state["classes"], probability.tolist(), strict=True)),
        "scope": "Source morphology population; no independent batch or unknown-class validation",
    }


if __name__ == "__main__":
    root = Path.cwd()
    print(
        json.dumps(
            predict(root, json.loads((root / "configs/example-input.json").read_text())), indent=2
        )
    )
