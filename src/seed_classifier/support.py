"""Training-only marginal ranges; no multivariate or unknown-variety guarantee."""

import numpy as np

from seed_classifier.data import FEATURES


def training_feature_ranges(training):
    """Record source-unit minima/maxima from the training partition only."""
    values = training[FEATURES].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Training support requires nonempty finite positive morphology")
    return {
        name: {"min": float(values[:, i].min()), "max": float(values[:, i].max())}
        for i, name in enumerate(FEATURES)
    }


def range_diagnostics(payload, ranges=None):
    """Flag extrapolation without modifying the classifier or its probabilities."""
    result = {
        "status": "unavailable",
        "out_of_range_features": [],
        "basis": "Training-partition marginal feature minima/maxima",
        "limitation": "Inside all ranges does not establish joint support, independent batch validity or unknown-variety rejection",
    }
    if ranges is None:
        return result
    if set(ranges) != set(FEATURES):
        raise ValueError("Training support metadata must cover the exact source feature schema")
    for name in FEATURES:
        low, high = ranges[name]["min"], ranges[name]["max"]
        if not np.isfinite([low, high]).all() or low <= 0 or low > high:
            raise ValueError("Invalid training support metadata")
        if payload[name] < low or payload[name] > high:
            result["out_of_range_features"].append(name)
    result["status"] = (
        "outside_training_ranges" if result["out_of_range_features"] else "within_training_ranges"
    )
    return result
