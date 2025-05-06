from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from student_grade_prediction.paths import (  # noqa: E402
    DATASET_PATH,
    FEATURE_IMPORTANCE_PATH,
    HOLDOUT_PREDICTIONS_PATH,
    MODEL_ARTIFACT_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_REPORT_PATH,
)
from student_grade_prediction.schema import (  # noqa: E402
    CATEGORY_LABELS,
    FIELD_GROUPS,
    FIELD_HELP,
    FIELD_LABELS,
    SLIDER_CONFIG,
)

st.set_page_config(
    page_title="Student Grade Forecast",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


@st.cache_data
def load_report() -> dict:
    return json.loads(MODEL_REPORT_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_comparison() -> pd.DataFrame:
    return pd.read_csv(MODEL_COMPARISON_PATH)


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


@st.cache_data
def load_holdout_predictions() -> pd.DataFrame:
    return pd.read_csv(HOLDOUT_PREDICTIONS_PATH)


@st.cache_resource
def load_model_bundle() -> dict:
    return joblib.load(MODEL_ARTIFACT_PATH)


def category_options(df: pd.DataFrame, column: str) -> list:
    values = df[column].dropna().tolist()
    unique_values = sorted(set(values))
    preferred_order = CATEGORY_LABELS.get(column, {})
    if preferred_order:
        ordered = [value for value in preferred_order if value in unique_values]
        extras = [value for value in unique_values if value not in preferred_order]
        return ordered + extras
    return unique_values


def default_values(df: pd.DataFrame) -> dict:
    defaults: dict[str, object] = {}
    for column in df.columns:
        if column == "G3":
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            defaults[column] = int(round(df[column].median()))
        else:
            defaults[column] = df[column].mode().iloc[0]
    return defaults


def render_field(df: pd.DataFrame, column: str, defaults: dict[str, object]):
    label = FIELD_LABELS.get(column, column)
    help_text = FIELD_HELP.get(column)

    if column in CATEGORY_LABELS:
        options = category_options(df, column)
        default = defaults[column]
        default_index = options.index(default) if default in options else 0
        return st.selectbox(
            label,
            options,
            index=default_index,
            format_func=lambda value: CATEGORY_LABELS[column].get(value, str(value)),
            help=help_text,
        )

    if column in SLIDER_CONFIG:
        config = SLIDER_CONFIG[column]
        default = int(defaults[column])
        min_value = int(config["min"])
        max_value = int(config["max"])
        default = min(max(default, min_value), max_value)
        return st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            value=default,
            step=int(config.get("step", 1)),
            help=help_text,
        )

    series = df[column]
    min_value = int(series.min())
    max_value = int(series.max())
    default = int(defaults[column])
    default = min(max(default, min_value), max_value)
    return st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=default,
        step=1,
        help=help_text,
    )


def build_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    defaults = default_values(df)
    feature_values: dict[str, object] = {}

    with st.form("prediction_form"):
        for section_name, fields in FIELD_GROUPS:
            with st.expander(section_name, expanded=section_name in {"Academic Record", "Student Profile"}):
                columns = st.columns(3)
                for index, field in enumerate(fields):
                    with columns[index % 3]:
                        feature_values[field] = render_field(df, field, defaults)

        submitted = st.form_submit_button("Predict Final Grade")

    if not submitted:
        return pd.DataFrame()

    return pd.DataFrame([feature_values])


def qualitative_band(prediction: float) -> str:
    if prediction >= 16:
        return "Excellent"
    if prediction >= 13:
        return "Strong"
    if prediction >= 10:
        return "Passing"
    return "At risk"


def predict_interval(pipeline, input_frame: pd.DataFrame) -> tuple[float, float]:
    preprocessed = pipeline.named_steps["preprocess"].transform(input_frame)
    forest = pipeline.named_steps["model"]
    tree_predictions = np.array([tree.predict(preprocessed)[0] for tree in forest.estimators_])
    lower, upper = np.percentile(tree_predictions, [10, 90])
    return float(np.clip(lower, 0, 20)), float(np.clip(upper, 0, 20))


def show_home_page(df: pd.DataFrame, report: dict):
    st.title("Student Grade Forecast")
    st.caption("Mid-term final grade forecasting using student context and prior period grades.")

    scenario = report["scenario"]
    holdout = report["holdout_metrics"]
    dataset = report["dataset"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Students", dataset["rows"])
    col2.metric("Features", dataset["feature_count"])
    col3.metric("Holdout MAE", f'{holdout["mae"]:.2f}')
    col4.metric("Holdout R²", f'{holdout["r2"]:.3f}')

    left, right = st.columns([1.1, 0.9])
    with left:
        st.subheader("Problem Setup")
        st.write(scenario["description"])
        st.markdown(
            "\n".join(
                [
                    f'- Target: `{dataset["target"]}`',
                    f'- Best model: `{report["selected_model"]}`',
                    f'- Cross-validation: `{report["cross_validation"]}`',
                    f'- Holdout split: `{report["holdout_split"]}`',
                ]
            )
        )

    with right:
        st.subheader("Dataset Snapshot")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.subheader("Intended Use")
    st.write(
        "This application is a reproducible machine learning demonstration for student-grade forecasting. "
        "It should be used as decision support only, not as the sole basis for academic intervention or evaluation."
    )

    st.subheader("Evaluation Summary")
    summary = pd.DataFrame(
        {
            "Metric": ["MAE", "RMSE", "R²"],
            "Cross-validation": [
                report["selected_model_cv"]["mae_mean"],
                report["selected_model_cv"]["rmse_mean"],
                report["selected_model_cv"]["r2_mean"],
            ],
            "Holdout": [holdout["mae"], holdout["rmse"], holdout["r2"]],
        }
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)


def show_evaluation_page(comparison: pd.DataFrame, holdout_predictions: pd.DataFrame):
    st.title("Model Evaluation")
    st.caption("All reported results use the same feature set and preprocessing pipeline.")

    display_df = comparison.copy()
    numeric_cols = [column for column in display_df.columns if column != "model"]
    display_df[numeric_cols] = display_df[numeric_cols].round(3)

    st.subheader("Cross-validated model comparison")
    st.dataframe(
        display_df.style.highlight_min(subset=["cv_mae_mean", "cv_rmse_mean"], color="#d9f2d9").highlight_max(
            subset=["cv_r2_mean"], color="#d9f2d9"
        ),
        use_container_width=True,
        hide_index=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(comparison["model"], comparison["cv_mae_mean"], color="#1f77b4")
        ax.set_title("Cross-validated MAE")
        ax.set_ylabel("Mean absolute error")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(comparison["model"], comparison["cv_r2_mean"], color="#2ca02c")
        ax.set_title("Cross-validated R²")
        ax.set_ylabel("R²")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

    st.subheader("Holdout predictions")
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(
        holdout_predictions["actual"],
        holdout_predictions["predicted"],
        alpha=0.7,
        color="#ff7f0e",
        edgecolors="black",
        linewidths=0.4,
    )
    ax.plot([0, 20], [0, 20], linestyle="--", color="gray")
    ax.set_xlabel("Actual final grade")
    ax.set_ylabel("Predicted final grade")
    ax.set_title("Holdout set: actual vs predicted")
    st.pyplot(fig)

    st.dataframe(holdout_predictions.head(15), use_container_width=True, hide_index=True)


def show_prediction_page(df: pd.DataFrame, model_bundle: dict):
    st.title("Predict Final Grade")
    st.caption("Fill in the student profile used by the trained model. The result is clipped to the 0-20 grading scale.")
    st.info(
        "Interpret forecasts alongside teacher judgment, attendance context, and qualitative student information. "
        "The prediction is an estimate from historical data, not a final verdict."
    )

    input_frame = build_input_frame(df)
    if input_frame.empty:
        st.info("Submit the form to generate a prediction.")
        return

    pipeline = model_bundle["model"]
    raw_prediction = float(pipeline.predict(input_frame)[0])
    prediction = float(np.clip(raw_prediction, 0, 20))
    lower, upper = predict_interval(pipeline, input_frame)

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted grade", f"{prediction:.2f} / 20")
    col2.metric("Performance band", qualitative_band(prediction))
    col3.metric("Ensemble range", f"{lower:.2f} to {upper:.2f}")
    st.progress(int(round((prediction / 20) * 100)))

    st.subheader("Submitted student profile")
    st.dataframe(input_frame, use_container_width=True, hide_index=True)


def show_interpretability_page(feature_importance: pd.DataFrame):
    st.title("Model Interpretation")
    st.caption("Permutation importance on the held-out test set. Larger values indicate stronger impact on predictive performance.")

    top_features = feature_importance.head(12)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(top_features["feature"], top_features["importance_mean"], color="#9467bd")
    ax.invert_yaxis()
    ax.set_xlabel("Importance (mean decrease in score)")
    ax.set_title("Top model drivers")
    st.pyplot(fig)

    st.dataframe(feature_importance, use_container_width=True, hide_index=True)


def main():
    required_artifacts = [
        DATASET_PATH,
        MODEL_ARTIFACT_PATH,
        MODEL_REPORT_PATH,
        MODEL_COMPARISON_PATH,
        FEATURE_IMPORTANCE_PATH,
        HOLDOUT_PREDICTIONS_PATH,
    ]
    missing = [path.name for path in required_artifacts if not path.exists()]
    if missing:
        st.error(
            "Missing model artifacts. Run `python train_model.py` from the project root before launching the app.\n\n"
            + "Missing files: "
            + ", ".join(missing)
        )
        st.stop()

    df = load_dataset()
    report = load_report()
    comparison = load_comparison()
    feature_importance = load_feature_importance()
    holdout_predictions = load_holdout_predictions()
    model_bundle = load_model_bundle()

    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Model Evaluation", "Predict Final Grade", "Interpretability"],
    )

    if page == "Overview":
        show_home_page(df, report)
    elif page == "Model Evaluation":
        show_evaluation_page(comparison, holdout_predictions)
    elif page == "Predict Final Grade":
        show_prediction_page(df, model_bundle)
    else:
        show_interpretability_page(feature_importance)


if __name__ == "__main__":
    main()
