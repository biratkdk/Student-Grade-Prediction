# Contributing

## Development Setup

1. Install dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

2. Train the model artifacts:

```bash
python train_model.py
```

3. Run tests:

```bash
python -m pytest -q
```

4. Launch the application:

```bash
streamlit run app.py
```

## Project Conventions

- Keep the training logic in `src/student_grade_prediction/training.py`
- Keep UI schema and field labels in `src/student_grade_prediction/schema.py`
- Do not retrain models inside the Streamlit UI
- Update documentation when the scenario, metrics, or repo layout changes

## Pull Request Checklist

- The code runs locally
- `python train_model.py` completes successfully
- `python -m pytest -q` passes
- README and model-facing docs are updated when behavior changes
- Generated artifacts are reviewed before release
