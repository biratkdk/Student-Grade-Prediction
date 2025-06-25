# Changelog

All notable changes to this project will be documented in this file.

## Project Timeline

- Late 2024: initial exploratory analysis, notebook experimentation, and first version of the grade prediction app
- 2026-04-06: repository refactor into a reproducible training and application workflow

## [0.1.0] - 2026-04-06

### Added

- Reproducible training entrypoint with saved model artifacts
- Standard sklearn preprocessing and model selection pipeline
- Cross-validated comparison across six regression models
- Tuned Random Forest final model
- Holdout prediction export and permutation feature importance
- Automated tests and GitHub Actions CI
- Model card, contributing guide, and Streamlit configuration

### Changed

- Reworked the Streamlit app to load a persisted model instead of training on demand
- Reorganized the repository into `data/`, `models/`, `notebooks/`, `src/`, and `tests/`
- Rewrote the README around one consistent prediction scenario
