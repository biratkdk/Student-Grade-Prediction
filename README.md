# Student Grade Prediction Dashboard

[![GitHub Profile](https://img.shields.io/badge/GitHub-biratkdk-blue?style=flat-square)](https://github.com/biratkdk)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](./LICENSE)

## Overview

An interactive web-based machine learning dashboard for predicting Portuguese student final grades. Built with Python and Streamlit to demonstrate full-stack data science capabilities.

## Project Goal

Predict final grades (0-20 scale) for Portuguese high school students using 32 demographic, social, and academic features.

## Quick Start

### Requirements
- Python 3.8+
- pip or conda

### Setup

1. Clone repository
   `ash
   git clone https://github.com/biratkdk/student-grade-prediction.git
   cd student-grade-prediction
   `

2. Create virtual environment
   `ash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   `

3. Install dependencies
   `ash
   pip install -r requirements.txt
   `

4. Run application
   `ash
   streamlit run app.py
   `

Open http://localhost:8501 in browser.

## Features

### Home Dashboard
- Project overview and quick statistics
- Dataset summary (395 students, 32 features)
- Key metrics at a glance

### Data Analysis
- Grade distribution visualization
- Top predictors chart
- Student demographics breakdown

### Grade Prediction
- Interactive sliders for student attributes
- Real-time ML predictions
- Performance feedback

### Model Comparison
- 6 ML algorithms: Linear Regression, ElasticNet, Random Forest, Extra Trees, SVM, Gradient Boosting
- Performance metrics (MAE, RMSE, R2)
- Visual comparisons

## Dataset

- **Size**: 395 students
- **Features**: 32 attributes
- **Target**: Final grade (G3, 0-20 scale)
- **Quality**: Zero missing values
- **Source**: UCI Machine Learning Repository

## Technology Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| Streamlit | Web framework |
| scikit-learn | ML models |
| pandas | Data processing |
| numpy | Numerical computing |
| matplotlib | Visualization |

## Model Performance

Train-test split: 75-25 (296 training, 99 test samples)

| Algorithm | MAE | RMSE | Status |
|-----------|-----|------|--------|
| Linear Regression | 3.4851 | 4.4326 | BEST |
| Gradient Boosting | 3.5721 | 4.5006 | 2nd |
| SVM | 3.5493 | 4.5815 | 3rd |
| ElasticNet | 3.6081 | 4.5733 | 4th |
| Random Forest | 3.6443 | 4.6273 | 5th |
| Extra Trees | 3.7793 | 4.7470 | 6th |
| Baseline | 3.7879 | 4.8252 | Reference |

Key: All ML models outperform baseline by 8%

## Project Structure

`
student-grade-prediction/
+-- app.py                          # Streamlit application
+-- student-mat.csv                 # Dataset
+-- Student Grade Analysis & Prediction.ipynb   # Notebook
+-- requirements.txt                # Dependencies
+-- README.md                       # This file
+-- LICENSE                         # MIT License
+-- .venv/                          # Virtual environment
`

## Implementation Notes

This is an original implementation built from scratch for a 5-credit university machine learning project. The project includes:
- Complete feature engineering and model selection
- Interactive web dashboard with Streamlit
- Real-time predictions with demographic filtering
- Comprehensive EDA and statistical analysis

Dataset sourced from UCI ML Repository (public domain).

## Author

**Birat Khadka**
- GitHub: [@biratkdk](https://github.com/biratkdk)
- College: United Technical College of Engineering

## License

MIT License - See LICENSE file for details

