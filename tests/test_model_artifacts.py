from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from student_grade_prediction.paths import (
    DATASET_PATH,
    FEATURE_IMPORTANCE_PATH,
    HOLDOUT_PREDICTIONS_PATH,
    MODEL_ARTIFACT_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_REPORT_PATH,
)


def test_required_artifacts_exist():
    for path in [
        DATASET_PATH,
        MODEL_ARTIFACT_PATH,
        MODEL_REPORT_PATH,
        MODEL_COMPARISON_PATH,
        FEATURE_IMPORTANCE_PATH,
        HOLDOUT_PREDICTIONS_PATH,
    ]:
        assert path.exists(), f"Missing artifact: {path}"


def test_model_bundle_predicts_single_row():
    bundle = joblib.load(MODEL_ARTIFACT_PATH)
    model = bundle["model"]

    df = pd.read_csv(DATASET_PATH)
    X = df.drop(columns=["G3"])
    prediction = model.predict(X.head(1))

    assert prediction.shape == (1,)
    assert np.isfinite(prediction[0])
    assert 0 <= prediction[0] <= 20


def test_report_and_comparison_are_consistent():
    report = json.loads(MODEL_REPORT_PATH.read_text(encoding="utf-8"))
    comparison = pd.read_csv(MODEL_COMPARISON_PATH)

    assert report["selected_model"] == "Random Forest"
    assert comparison.iloc[0]["model"] == "Random Forest"
    assert comparison["cv_mae_mean"].is_monotonic_increasing


def test_feature_importance_and_holdout_predictions_are_not_empty():
    feature_importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    holdout_predictions = pd.read_csv(HOLDOUT_PREDICTIONS_PATH)

    assert not feature_importance.empty
    assert not holdout_predictions.empty
    assert {"actual", "predicted", "residual"} <= set(holdout_predictions.columns)
