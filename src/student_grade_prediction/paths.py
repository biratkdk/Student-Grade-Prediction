from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

DATASET_PATH = DATA_DIR / "student-mat.csv"
MODEL_ARTIFACT_PATH = MODELS_DIR / "student_grade_model.joblib"
MODEL_REPORT_PATH = MODELS_DIR / "model_report.json"
MODEL_COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"
FEATURE_IMPORTANCE_PATH = MODELS_DIR / "feature_importance.csv"
HOLDOUT_PREDICTIONS_PATH = MODELS_DIR / "holdout_predictions.csv"
