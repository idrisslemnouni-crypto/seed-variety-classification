"""Fixed model comparison, validation-only selection and temperature calibration."""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from seed_classifier.data import FEATURES, read_source, split_data


def scores(model, x):
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return np.log(np.clip(model.predict_proba(x), 1e-8, 1))


def fit_temperature(logits, y, classes):
    result = minimize_scalar(
        lambda t: log_loss(y, softmax(logits / t, axis=1), labels=classes),
        bounds=(0.05, 10),
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not result.success:
        raise ValueError("Validation temperature fit failed")
    return float(result.x)


def probability_metrics(y, probabilities, classes):
    labels = np.array([list(classes).index(value) for value in y])
    onehot = np.eye(len(classes))[labels]
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    ece = 0.0
    for i in range(10):
        mask = (confidence >= i / 10) & (confidence < (i + 1) / 10 if i < 9 else confidence <= 1)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return {
        "log_loss": float(log_loss(y, probabilities, labels=classes)),
        "multiclass_brier": float(np.mean(np.sum((probabilities - onehot) ** 2, axis=1))),
        "ece_10_bins": float(ece),
    }


def run(root):
    config = json.loads((root / "configs/default.json").read_text())
    frame, audit = read_source(root)
    train, validation, test = split_data(frame, config["seed"])
    methods = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic": make_pipeline(
            StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=42)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=2,
        ),
        "rbf_svm": make_pipeline(
            StandardScaler(), SVC(C=10, gamma="scale", decision_function_shape="ovr")
        ),
    }
    validation_scores = {}
    for name, model in methods.items():
        model.fit(train[FEATURES], train.Class)
        validation_scores[name] = float(
            f1_score(validation.Class, model.predict(validation[FEATURES]), average="macro")
        )
    selected = max(validation_scores, key=validation_scores.get)
    model = methods[selected]
    classes = model.classes_.tolist()
    temperature = fit_temperature(scores(model, validation[FEATURES]), validation.Class, classes)
    # Selected method and temperature are now frozen before test evaluation.
    comparisons = []
    for name, candidate in methods.items():
        predicted = candidate.predict(test[FEATURES])
        comparisons.append(
            {
                "method": name,
                "validation_macro_f1": validation_scores[name],
                "test_macro_f1": float(f1_score(test.Class, predicted, average="macro")),
                "test_accuracy": float(accuracy_score(test.Class, predicted)),
            }
        )
    predicted = model.predict(test[FEATURES])
    raw = softmax(scores(model, test[FEATURES]), axis=1)
    calibrated = softmax(scores(model, test[FEATURES]) / temperature, axis=1)
    reports = root / "reports"
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    evidence = {
        "source_audit": audit,
        "config": config,
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "class_counts": frame.Class.value_counts().sort_index().to_dict(),
        "selected_on_validation": selected,
        "temperature_fit_on_validation": temperature,
        "comparison": comparisons,
        "test_probability_metrics": {
            "uncalibrated_margin_softmax": probability_metrics(test.Class, raw, classes),
            "calibrated": probability_metrics(test.Class, calibrated, classes),
        },
        "voting_vs_probability_argmax_mismatches": int(
            np.sum(predicted != np.asarray(classes)[calibrated.argmax(axis=1)])
        ),
        "classification_report": classification_report(test.Class, predicted, output_dict=True),
        "scope": "Random split of unique source grains; batch/farm independence unavailable; no raw-image segmentation",
    }
    (reports / "metrics.json").write_text(json.dumps(evidence, indent=2, allow_nan=False))
    pd.DataFrame(comparisons).to_csv(reports / "comparison.csv", index=False)
    output = test[["source_row_id", "Class"]].rename(columns={"Class": "actual"}).copy()
    output["predicted"] = predicted
    for i, label in enumerate(classes):
        output["probability_" + label] = calibrated[:, i]
    output.to_csv(reports / "test-predictions.csv", index=False)
    membership = pd.concat(
        [
            part[["source_row_id", "Class"]].assign(split=name)
            for name, part in [("train", train), ("validation", validation), ("test", test)]
        ]
    )
    membership.to_csv(reports / "split-membership.csv", index=False)
    (root / "models").mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "temperature": temperature,
            "features": FEATURES,
            "classes": classes,
            "selected": selected,
        },
        root / "models/selected.joblib",
    )
    (root / "configs/example-input.json").write_text(
        json.dumps(test[FEATURES].iloc[0].to_dict(), indent=2)
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        test.Class,
        predicted,
        labels=classes,
        ax=ax,
        colorbar=False,
        cmap="Greens",
        xticks_rotation=45,
    )
    ax.set_title(
        "Unique-grain held-out confusion matrix\nRandom split; no independent batch validation"
    )
    fig.tight_layout()
    fig.savefig(reports / "figures/confusion-matrix.png", dpi=150)
    plt.close(fig)
    importance = permutation_importance(
        model,
        test[FEATURES],
        test.Class,
        n_repeats=5,
        random_state=42,
        scoring="f1_macro",
        n_jobs=2,
    )
    ranking = pd.DataFrame(
        {
            "feature": FEATURES,
            "mean_macro_f1_drop": importance.importances_mean,
            "std": importance.importances_std,
        }
    ).sort_values("mean_macro_f1_drop")
    ranking.to_csv(reports / "permutation-importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(ranking.feature, ranking.mean_macro_f1_drop, xerr=ranking["std"], color="#236b4e")
    ax.set(
        xlabel="Held-out macro-F1 drop (five permutations)",
        title="Post-hoc sensitivity · correlated geometry limits attribution",
    )
    fig.tight_layout()
    fig.savefig(reports / "figures/permutation-importance.png", dpi=150)
    plt.close(fig)
    return evidence


if __name__ == "__main__":
    result = run(Path.cwd())
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "source_audit",
                    "split_rows",
                    "selected_on_validation",
                    "temperature_fit_on_validation",
                    "comparison",
                    "test_probability_metrics",
                ]
            },
            indent=2,
        )
    )
