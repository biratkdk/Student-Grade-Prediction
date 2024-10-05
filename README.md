# Student Grade Prediction Dashboard

[![GitHub Profile](https://img.shields.io/badge/GitHub-biratkdk-blue?style=flat-square)](https://github.com/biratkdk)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](./LICENSE)

## Overview

An interactive web-based machine learning dashboard for predicting Portuguese student final grades. Built with Python and Streamlit to demonstrate full-stack data science capabilities including data analysis, model development, and web application design.

---

## Project Goal

Predict final grades (0-20 scale) for Portuguese high school students using 32 demographic, social, and academic features. The application includes exploratory data analysis, real-time predictions, and multi-model performance comparison.

---

## Quick Start

### Requirements
- Python 3.8+
- pip or conda

### Setup

1. Clone the repository
   \\\ash
   git clone https://github.com/biratkdk/student-grade-prediction.git
   cd student-grade-prediction
   \\\

2. Create virtual environment
   \\\ash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   \\\

3. Install dependencies
   \\\ash
   pip install -r requirements.txt
   \\\

4. Run the application
   \\\ash
   streamlit run app.py
   \\\

Open http://localhost:8502 in your browser.

---

## Features

### 🏠 Home Dashboard
- Project overview with quick statistics
- Dataset summary (395 students, 32 features)
- Key metrics at a glance

### 📊 Data Analysis
- Interactive grade distribution visualization
- Correlation analysis of top predictive features
- Student demographics breakdown
- Data patterns and insights

### 🔮 Grade Prediction
- Interactive input sliders for student attributes
- Real-time ML predictions
- Performance feedback (Excellent/Good/Average/Needs Improvement)
- Supports personalized predictions

### 📈 Model Comparison
- 3 Machine Learning algorithms:
  - Linear Regression
  - Random Forest
  - Gradient Boosting
- Performance metrics (MAE, RMSE, R²)
- Visual model comparisons

---

## Dataset

- **Size**: 395 students
- **Features**: 32 attributes
- **Target**: Final grade (G3, 0-20 scale)
- **Categories**: Demographics, parent background, school info, academic performance, social habits
- **Source**: UCI Machine Learning Repository

---

## Technology Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| Streamlit | Web framework |
| scikit-learn | ML models |
| pandas | Data processing |
| numpy | Numerical computing |
| matplotlib | Visualization |

---

## Model Performance

Train-test split: 75-25 with random_state=42

| Algorithm | MAE | RMSE | R² |
|-----------|-----|------|-----|
| Linear Regression | 2.45 | 3.12 | 0.31 |
| Random Forest | 2.38 | 3.05 | 0.35 |
| Gradient Boosting | 2.40 | 3.08 | 0.34 |

Linear Regression provides the best balance of accuracy and simplicity.

---

## Project Structure

\\\
student-grade-prediction/
├── app.py                    # Main Streamlit application
├── student-mat.csv           # Dataset
├── Student Grade Analysis & Prediction.ipynb  # Analysis notebook
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── LICENSE                   # MIT License
└── .gitignore               # Git configuration
\\\

---

## Skills Demonstrated

- **Data Analysis**: EDA, correlation analysis, statistical insights
- **Machine Learning**: Model training, evaluation, hyperparameter tuning
- **Data Engineering**: Data preprocessing, categorical encoding, feature selection
- **Web Development**: Interactive web application with Streamlit
- **Python Development**: Clean code practices, efficient implementations
- **Version Control**: Git workflow and repository management

---

## License

MIT License - open source and free to use.

---

## Author

**Birat Khadka**
- GitHub: [@biratkdk](https://github.com/biratkdk)
- Location: United Technical College of Engineering
