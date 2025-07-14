# Machine Learning Engineer Assignment

## Overview
This repository contains a complete machine learning pipeline for predicting DON concentration (vomitoxin_ppb) in corn samples using hyperspectral imaging data. The project covers data preprocessing, exploratory data analysis (EDA), model training and evaluation, and model interpretability using SHAP. The solution is designed to be modular, production-ready, and easy to extend.

## Directory Structure
```
ML_Engineer_Assignment/
├── data/
│   └── MLE-Assignment.csv         # Hyperspectral dataset file
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
```

## Installation
1. **Clone the Repository:**
   ```bash 
   git clone https://github.com/Krish-Lakhani19/ML_Engineer_Assignment
   cd ML_Engineer_Assignment
   ```

2. **Create and Activate a Virtual Environment:**
   - On macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Running the Machine Learning Pipeline
Execute the pipeline to load data, preprocess, visualize, train, evaluate, and interpret the model:
```bash
python src/ml_pipeline.py
```
This script will also save the trained model and scaler in the `models/` directory.

### Running the Unit Tests
To run the unit tests, execute:
```bash
python -m unittest discover tests
```
Ensure you run this from the project root so that Python can locate the `src` package correctly.

## Report
A comprehensive project report detailing the methodology, procedures, outputs, and model performance is available in `docs/report.pdf`.

## Contact
For questions or further information, please contact [krishlakhani46767@gmail.com].
