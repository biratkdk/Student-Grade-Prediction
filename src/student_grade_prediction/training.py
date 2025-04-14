from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from .paths import (
    DATASET_PATH,
    FEATURE_IMPORTANCE_PATH,
    HOLDOUT_PREDICTIONS_PATH,
    MODEL_ARTIFACT_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_REPORT_PATH,
    MODELS_DIR,
)

TARGET_COLUMN = "G3"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str


SCENARIO = Scenario(
    name="Mid-term final grade forecast",
    description=(
        "Predict the final grade (G3) using all student attributes available before the final result, "
        "including first period grade (G1) and second period grade (G2)."
    ),
)


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def build_candidate_models() -> dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "ElasticNet": ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
        "Extra Trees": ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE),
        "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.2, gamma="scale"),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def build_pipeline(X: pd.DataFrame, model) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X)),
            ("model", model),
        ]
    )


def evaluate_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    rows = []
    for name, model in build_candidate_models().items():
        pipeline = build_pipeline(X, model)
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        rows.append(
            {
                "model": name,
                "cv_mae_mean": -scores["test_mae"].mean(),
                "cv_mae_std": scores["test_mae"].std(),
                "cv_rmse_mean": -scores["test_rmse"].mean(),
                "cv_rmse_std": scores["test_rmse"].std(),
                "cv_r2_mean": scores["test_r2"].mean(),
                "cv_r2_std": scores["test_r2"].std(),
            }
        )

    comparison = pd.DataFrame(rows).sort_values(by="cv_mae_mean", ascending=True).reset_index(drop=True)
    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    return comparison


def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pipeline = build_pipeline(X_train, RandomForestRegressor(random_state=RANDOM_STATE))
    parameter_space = {
        "model__n_estimators": randint(250, 650),
        "model__max_depth": [None, 8, 12, 16, 20],
        "model__min_samples_split": randint(2, 15),
        "model__min_samples_leaf": randint(1, 8),
        "model__max_features": ["sqrt", "log2", None],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_space,
        n_iter=20,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search


def save_feature_importance(estimator: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    importance = permutation_importance(
        estimator,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=20,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    importance_df = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values(by="importance_mean", ascending=False)
    importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    return importance_df


def save_holdout_predictions(y_test: pd.Series, predictions: np.ndarray) -> pd.DataFrame:
    holdout_df = pd.DataFrame(
        {
            "actual": y_test.values,
            "predicted": predictions,
            "residual": y_test.values - predictions,
        }
    )
    holdout_df.to_csv(HOLDOUT_PREDICTIONS_PATH, index=False)
    return holdout_df


def fit_final_model(X: pd.DataFrame, y: pd.Series, best_params: dict[str, object]) -> Pipeline:
    final_model = RandomForestRegressor(random_state=RANDOM_STATE, **best_params)
    final_pipeline = build_pipeline(X, final_model)
    final_pipeline.fit(X, y)
    return final_pipeline


def save_model_bundle(model: Pipeline, feature_names: list[str]):
    bundle = {
        "model": model,
        "feature_names": feature_names,
        "target": TARGET_COLUMN,
        "scenario": SCENARIO.name,
    }
    joblib.dump(bundle, MODEL_ARTIFACT_PATH)


def normalize_param_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def save_report(
    df: pd.DataFrame,
    comparison: pd.DataFrame,
    search: RandomizedSearchCV,
    holdout_metrics: dict[str, float],
):
    selected_row = comparison.loc[comparison["model"] == "Random Forest"].iloc[0]
    report = {
        "scenario": {
            "name": SCENARIO.name,
            "description": SCENARIO.description,
        },
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "feature_count": int(df.shape[1] - 1),
            "target": TARGET_COLUMN,
        },
        "cross_validation": f"{CV_SPLITS}-fold KFold with shuffle=True and random_state={RANDOM_STATE}",
        "holdout_split": f"{int(TEST_SIZE * 100)}% test split with random_state={RANDOM_STATE}",
        "selected_model": "Random Forest",
        "selected_model_cv": {
            "mae_mean": float(selected_row["cv_mae_mean"]),
            "mae_std": float(selected_row["cv_mae_std"]),
            "rmse_mean": float(selected_row["cv_rmse_mean"]),
            "rmse_std": float(selected_row["cv_rmse_std"]),
            "r2_mean": float(selected_row["cv_r2_mean"]),
            "r2_std": float(selected_row["cv_r2_std"]),
        },
        "best_params": {
            key.replace("model__", ""): normalize_param_value(value)
            for key, value in search.best_params_.items()
        },
        "holdout_metrics": holdout_metrics,
    }
    MODEL_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    X, y = split_features_and_target(df)

    comparison = evaluate_models(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    search = tune_random_forest(X_train, y_train)
    best_estimator = search.best_estimator_
    predictions = np.clip(best_estimator.predict(X_test), 0, 20)
    holdout_metrics = {
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "r2": float(r2_score(y_test, predictions)),
    }

    save_holdout_predictions(y_test, predictions)
    save_feature_importance(best_estimator, X_test, y_test)

    final_params = {
        key.replace("model__", ""): normalize_param_value(value)
        for key, value in search.best_params_.items()
    }
    final_pipeline = fit_final_model(X, y, final_params)
    save_model_bundle(final_pipeline, X.columns.tolist())
    save_report(df, comparison, search, holdout_metrics)

    print("Training complete")
    print("Selected model: Random Forest")
    print(
        "Holdout metrics: "
        f"MAE={holdout_metrics['mae']:.3f}, "
        f"RMSE={holdout_metrics['rmse']:.3f}, "
        f"R2={holdout_metrics['r2']:.3f}"
    )


if __name__ == "__main__":
    main()
