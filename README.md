# Student Grade Forecast

Streamlit application and training pipeline for forecasting Portuguese secondary school students' final grades (`G3`) from academic, demographic, family, and lifestyle variables.

## Project Scope

This project uses a single, consistent prediction scenario:

- Forecast the final grade `G3`
- Use all variables available before the final result
- Include `G1` and `G2` as mid-term academic signals
- Train and evaluate the same pipeline that the app uses for inference

## What Changed

- Replaced ad hoc training inside the Streamlit app with a saved model artifact
- Standardized preprocessing with `Pipeline` and `ColumnTransformer`
- Added cross-validated comparison across six candidate models
- Tuned the final `RandomForestRegressor` and saved evaluation artifacts
- Added holdout predictions, feature importance, tests, and CI
- Reorganized the project into `data/`, `models/`, `notebooks/`, `src/`, and `tests/`

## Project Status

- Status: active and reproducible
- Initial project work: late 2024
- Major reproducibility and app refactor: April 2026
- Primary training script: `train_model.py`
- Primary app entrypoint: `app.py`
- Supporting docs: `MODEL_CARD.md`, `CONTRIBUTING.md`, `CHANGELOG.md`

## Repository Layout

```text
Student-Grade-Prediction-master/
|-- app.py
|-- train_model.py
|-- data/
|   `-- student-mat.csv
|-- models/
|   |-- feature_importance.csv
|   |-- holdout_predictions.csv
|   |-- model_comparison.csv
|   |-- model_report.json
|   `-- student_grade_model.joblib
|-- notebooks/
|   |-- README.md
|   `-- student_grade_analysis.ipynb
|-- src/
|   `-- student_grade_prediction/
|       |-- __init__.py
|       |-- paths.py
|       |-- schema.py
|       `-- training.py
|-- tests/
|   |-- conftest.py
|   `-- test_model_artifacts.py
|-- .github/
|   `-- workflows/
|       `-- ci.yml
|-- requirements.txt
|-- requirements-dev.txt
`-- README.md
```

## Dataset

- Source: UCI Student Performance dataset
- File: `data/student-mat.csv`
- Rows: 395
- Predictors used: 32
- Target: `G3`
- Missing values: 0

## Modeling Workflow

1. Load the raw dataset from `data/student-mat.csv`
2. Split predictors and target (`G3`)
3. Apply:
   - `StandardScaler` to numeric features
   - `OneHotEncoder(handle_unknown="ignore")` to categorical features
4. Compare six models with 5-fold cross-validation:
   - Linear Regression
   - ElasticNet
   - Random Forest
   - Extra Trees
   - SVR
   - Gradient Boosting
5. Tune the final Random Forest with `RandomizedSearchCV`
6. Evaluate on a held-out test split
7. Save:
   - trained model bundle
   - model comparison table
   - holdout predictions
   - permutation feature importance
   - model report JSON

## Running the Project

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

### 2. Train and generate artifacts

```bash
python train_model.py
```

### 3. Launch the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` after Streamlit starts.

## App Pages

- `Overview`: scenario definition, dataset summary, and evaluation snapshot
- `Model Evaluation`: cross-validated model comparison and holdout prediction plot
- `Predict Final Grade`: interactive form for generating a forecast
- `Interpretability`: permutation feature importance on the holdout set

## Limitations and Responsible Use

- This model is a statistical forecast, not a causal explanation of student performance.
- The data comes from a single public dataset and may not generalize to other institutions.
- Because `G1` and `G2` are included, this is a mid-term forecast rather than a cold-start prediction task.
- The app should be treated as decision support only, not as a replacement for teacher review or institutional policy.

## Tests

Run the local test suite after training:

```bash
python -m pytest -q
```

The tests check that:

- required artifacts exist
- the saved model predicts successfully
- the report and model comparison outputs are consistent
- feature importance and holdout prediction files are populated

## Continuous Integration

GitHub Actions workflow: `.github/workflows/ci.yml`

The CI job:

- installs dependencies
- runs `python train_model.py`
- runs `python -m pytest -q`

## Additional Documentation

- `MODEL_CARD.md`: model details, intended use, limitations, and latest metrics
- `CONTRIBUTING.md`: setup and development workflow
- `CHANGELOG.md`: notable project changes by version/date

## Notes

- Initial exploration for this project began in late 2024. The current repository structure, reproducible training pipeline, tests, and deployment-oriented app refinements were completed in 2026.
- The authoritative training logic is in `src/student_grade_prediction/training.py`
- The notebook in `notebooks/` is exploratory; the app does not depend on notebook state
- Predictions are clipped to the 0-20 grading scale for display consistency

## License

See `LICENSE`.
