# Machine Learning Engineer Assignment

## Overview
This repository contains a complete machine learning pipeline for predicting DON concentration (vomitoxin\_ppb) in corn samples using hyperspectral imaging data. The project includes data preprocessing, visualization, model training, evaluation, interpretability analysis using SHAP, and deployment integration via a Streamlit app.

## Features
- **Data Preprocessing:**  
  Handles missing values, normalizes feature data, and drops non-numeric columns (except the target).
- **Exploratory Data Analysis (EDA):**  
  Generates overlaid histograms, boxplots, line plots of average reflectance, and correlation heatmaps—all in a single consolidated figure.
- **Model Training and Evaluation:**  
  Trains a neural network regression model with early stopping. Evaluates performance using MAE, RMSE, and R² metrics along with diagnostic plots.
- **Model Interpretability:**  
  Uses SHAP to interpret model predictions and visualize feature importance.
- **Deployment-Ready:**  
  The trained model and scaler are saved for deployment, and a Streamlit app is provided for both manual and batch prediction via CSV upload.

## Directory Structure
ML_Engineer_Assignment/
├── data/
│   └── MLE-Assignment.csv         # The hyperspectral dataset
├── docs/
│   ├── README.md                  # Project overview, installation instructions, and usage details
│   └── report.pdf                 # Final compiled PDF report (or report.tex for LaTeX source)
├── models/
│   ├── my_model.keras             # Trained Keras model (saved in .keras or .h5 format)
│   └── saved_scaler.pkl           # Fitted scaler saved using pickle
├── src/
│   ├── __init__.py                # Marks src as a Python package (can be empty)
│   └── ml_pipeline.py             # Complete ML pipeline code (data loading, preprocessing, EDA, model training, evaluation, and interpretation)
├── tests/
│   └── test_pipeline.py           # Unit tests for functions in ml_pipeline.py
├── Dockerfile                     # Docker configuration for containerization (if needed)
└── requirements.txt               # List of required Python packages

## Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ML_Engineer_Assignment.git
   cd ML_Engineer_Assignment
2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # For macOS/Linux
   # or on Windows:
   venv\Scripts\activate
3.**Install the required packages**
  ```bash
 pip install -r requirements.txt

