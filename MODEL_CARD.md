# Model Card

## Model Summary

- Model type: `RandomForestRegressor`
- Target: `G3` final grade
- Training entrypoint: `python train_model.py`
- Training code: `src/student_grade_prediction/training.py`
- Inference app: `app.py`

## Prediction Scenario

The model forecasts the final grade (`G3`) using attributes available before the final result is known, including:

- demographic attributes
- family and support context
- study and attendance signals
- lifestyle variables
- first and second period grades (`G1`, `G2`)

## Training Data

- Dataset: UCI Student Performance, Portuguese language course subset
- File used here: `data/student-mat.csv`
- Rows: 395
- Missing values in source file: 0
- Target range: 0 to 20

## Preprocessing

- Numeric features: median imputation + standard scaling
- Categorical features: most-frequent imputation + one-hot encoding
- Cross-validation: 5-fold KFold with shuffle and `random_state=42`
- Holdout split: 20%

## Candidate Models Compared

- Linear Regression
- ElasticNet
- Random Forest
- Extra Trees
- SVR
- Gradient Boosting

## Selected Model

The current production artifact uses a tuned Random Forest.

Best parameters from the latest run:

- `max_depth = 12`
- `max_features = None`
- `min_samples_leaf = 5`
- `min_samples_split = 5`
- `n_estimators = 609`

## Performance

Cross-validated Random Forest:

- MAE: `1.023`
- RMSE: `1.594`
- R2: `0.870`

Holdout test split:

- MAE: `1.166`
- RMSE: `1.936`
- R2: `0.817`

These values come from the locally generated `models/model_report.json`.

## Intended Use

- Classroom analytics demonstrations
- Reproducible tabular ML workflow examples
- Exploratory forecasting of student outcomes
- Comparing how structured student attributes affect grade prediction

## Out-of-Scope Use

This model should not be used as:

- the sole basis for student evaluation
- an admissions or scholarship decision system
- a disciplinary decision engine
- a substitute for teacher judgment, counseling, or institutional policy

## Limitations

- The dataset is small and from a single educational context.
- Correlations in this dataset may not transfer to other schools or countries.
- Including `G1` and `G2` improves accuracy, but it also means this is a mid-term forecast, not an early-warning model.
- The model captures historical patterns, not causal relationships.
- Performance can degrade on populations unlike the training data.

## Ethical Considerations

- Student data is sensitive and context-dependent.
- Predictions can amplify existing inequities if used without human review.
- Qualitative context, teacher feedback, and institutional safeguards should always accompany model outputs.

## Maintenance

When the data or training code changes:

1. Run `python train_model.py`
2. Run `python -m pytest -q`
3. Review `models/model_report.json`
4. Re-check the app with `streamlit run app.py`
