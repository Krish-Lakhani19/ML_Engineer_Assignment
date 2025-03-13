# ml_pipeline.py
"""
Machine Learning Pipeline for Predicting DON Concentration (vomitoxin_ppb)
-----------------------------------------------------------------------------
This script implements a complete machine learning pipeline to:
  1. Load a hyperspectral imaging dataset from a CSV file.
  2. Preprocess the data: handle missing values, drop non-numeric columns,
     and normalize feature values.
  3. Visualize the data with a single consolidated figure containing:
       - Overlaid histograms for each feature.
       - A boxplot for the target variable (vomitoxin_ppb).
       - A line plot of the average reflectance (mean value) across features.
       - A heatmap of the correlation between features.
  4. Build and compile a simple neural network regression model.
  5. Train the model with early stopping.
  6. Evaluate the model using common regression metrics and diagnostic plots.
  7. Interpret the model predictions using SHAP.

Make sure to update the CSV file path and adjust the number of features if needed.

Author: [Krish Lakhani]
Date: [March 15th, 2025]
"""

import os
import logging
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# TensorFlow / Keras for building and training the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# SHAP for model interpretability
import shap

# Configure logging with time stamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Parameters:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset with columns stripped of extra whitespace.
    """
    logging.info("Loading dataset from %s", filepath)
    df = pd.read_csv(filepath)
    # Remove leading/trailing spaces in column names (if any)
    df.columns = df.columns.str.strip()
    logging.info("Data shape: %s", df.shape)
    logging.info("Data columns: %s", df.columns.tolist())
    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Preprocess the data:
      - Drop non-numeric columns (except the target variable).
      - Handle missing values by median imputation.
      - Normalize the feature columns using StandardScaler.

    Parameters:
        df (pd.DataFrame): The raw dataset.

    Returns:
        Tuple[pd.DataFrame, StandardScaler]: Preprocessed DataFrame and fitted scaler.
    """
    logging.info("Starting data preprocessing.")

    target_col = 'vomitoxin_ppb'

    # If target column is not numeric, raise an error
    if target_col not in df.columns:
        raise ValueError(f"Target variable '{target_col}' not found in dataset")

    # Select only numeric columns for features (preserving target even if non-numeric is not expected)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target_col not in numeric_cols:
        numeric_cols.append(target_col)
    df = df[numeric_cols]

    # Handle missing values using median imputation
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.median())
        logging.info("Missing values filled using median imputation.")
    else:
        logging.info("No missing values detected.")

    # Separate features and target
    features = df.drop(columns=[target_col])
    target = df[target_col]

    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Reassemble DataFrame with scaled features and original target
    df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    df_scaled[target_col] = target.values

    logging.info("Preprocessing complete. Data is normalized and ready.")
    return df_scaled, scaler


def visualize_data(df: pd.DataFrame, target_col: str = 'vomitoxin_ppb') -> None:
    """
    Create a single figure containing 4 subplots:
      1. Overlaid histograms for each feature (excluding the target).
      2. A boxplot for the target variable.
      3. A line plot of the average reflectance across features.
      4. A heatmap of correlations among features.

    This consolidated view provides a comprehensive summary of the dataset.

    Parameters:
        df (pd.DataFrame): The preprocessed dataset.
        target_col (str): The target column name.
    """
    logging.info("Creating a single figure with multiple visualizations.")

    # Prepare 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # --- Subplot 1: Overlaid Histograms for Each Feature ---
    features = [col for col in df.columns if col != target_col]
    for feature in features:
        axes[0, 0].hist(df[feature], bins=30, alpha=0.5, label=feature, edgecolor='black')
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Overlaid Histograms of Each Feature")
    axes[0, 0].legend(title="Features", fontsize=9)
    axes[0, 0].grid(True)

    # --- Subplot 2: Boxplot for the Target Variable ---
    sns.boxplot(x=df[target_col], ax=axes[0, 1])
    axes[0, 1].set_title(f"Boxplot of {target_col}")
    axes[0, 1].set_xlabel(target_col)

    # --- Subplot 3: Line Plot for Average Reflectance ---
    # Calculate the mean of each feature (excluding the target)
    avg_reflectance = df[features].mean()
    # Attempt to convert index to numeric if possible (e.g., wavelengths)
    try:
        x_values = pd.to_numeric(avg_reflectance.index, errors='coerce')
    except Exception:
        x_values = avg_reflectance.index
    axes[1, 0].plot(x_values, avg_reflectance.values, marker='o', linestyle='-')
    axes[1, 0].set_title("Average Spectral Reflectance")
    axes[1, 0].set_xlabel("Feature (Wavelength)")
    axes[1, 0].set_ylabel("Average Reflectance")
    axes[1, 0].tick_params(axis='x', rotation=45)

    # --- Subplot 4: Heatmap of Feature Correlations ---
    corr = df[features].corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', ax=axes[1, 1])
    axes[1, 1].set_title("Feature Correlation Heatmap")

    plt.tight_layout()
    plt.show()


def build_model(input_dim: int) -> Sequential:
    """
    Build and compile a simple neural network regression model.

    The model consists of two hidden layers with dropout regularization.

    Parameters:
        input_dim (int): Number of input features.

    Returns:
        Sequential: Compiled Keras model.
    """
    logging.info("Building the neural network model with input dimension %d", input_dim)
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # Output layer for regression
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    logging.info("Model built and compiled successfully.")
    return model


def train_model(model: Sequential, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray, epochs: int = 100, batch_size: int = 32) -> None:
    """
    Train the model using early stopping based on validation loss.

    Parameters:
        model (Sequential): Compiled Keras model.
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training target data.
        X_val (np.ndarray): Validation feature data.
        y_val (np.ndarray): Validation target data.
        epochs (int): Maximum number of epochs.
        batch_size (int): Batch size for training.
    """
    logging.info("Starting model training.")
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, callbacks=[early_stop], verbose=1)

    # Plot training and validation loss over epochs
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    logging.info("Model training complete.")


def evaluate_model(model: Sequential, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Evaluate the trained model using regression metrics and diagnostic plots.

    Metrics include Mean Absolute Error (MAE), RMSE, and R² score.
    Additionally, the function plots actual vs. predicted values and the residual distribution.

    Parameters:
        model (Sequential): Trained Keras model.
        X_test (np.ndarray): Test feature data.
        y_test (np.ndarray): Test target data.
    """
    logging.info("Evaluating model performance on test data.")
    predictions = model.predict(X_test).flatten()

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    logging.info("Evaluation Metrics - MAE: %.4f, RMSE: %.4f, R2: %.4f", mae, rmse, r2)
    print("Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Scatter plot: Actual vs Predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.7)
    plt.xlabel('Actual Target')
    plt.ylabel('Predicted Target')
    plt.title('Actual vs Predicted Values')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.show()

    # Histogram of residuals
    residuals = y_test - predictions
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True)
    plt.title('Residual Distribution')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.show()


def interpret_model(model: Sequential, X_sample: np.ndarray) -> None:
    """
    Use SHAP to interpret the model's predictions.

    A KernelExplainer is used on a subset of the data to generate SHAP values,
    and a summary plot is displayed to visualize feature importance.

    Parameters:
        model (Sequential): Trained Keras model.
        X_sample (np.ndarray): A sample of the feature data for SHAP analysis.
    """
    logging.info("Interpreting model predictions with SHAP.")
    # Use a subset of the data (e.g., first 100 samples) for faster SHAP computation
    explainer = shap.KernelExplainer(model.predict, X_sample[:100])
    shap_values = explainer.shap_values(X_sample[:100])
    shap.summary_plot(shap_values, X_sample[:100],
                      feature_names=[f"Feature {i + 1}" for i in range(X_sample.shape[1])])


def run_pipeline(data_filepath: str) -> None:
    """
    Run the entire machine learning pipeline:
      1. Load the data.
      2. Preprocess the data.
      3. Visualize the data in a single consolidated figure.
      4. Split the data into training, validation, and test sets.
      5. Build, train, and evaluate the model.
      6. Interpret the model using SHAP.

    Parameters:
        data_filepath (str): The path to the CSV dataset.
    """
    # Load and preprocess data
    df = load_data(data_filepath)
    df_processed, scaler = preprocess_data(df)

    # Visualize the dataset (all plots on a single page)
    visualize_data(df_processed, target_col='vomitoxin_ppb')

    # Prepare features and target arrays
    X = df_processed.drop(columns=['vomitoxin_ppb']).values
    y = df_processed['vomitoxin_ppb'].values

    # Split data: 80% training (with 10% of that as validation) and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # Build and train the model
    model = build_model(input_dim=X.shape[1])
    train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32)

    # Evaluate model performance on the test set
    evaluate_model(model, X_test, y_test)

    # Interpret the model using a subset of training data
    # Save the trained model and scaler for future use
    import os
    os.makedirs('models', exist_ok=True)  # Ensure the 'models' folder exists
    model.save('models/my_model.keras')

    import pickle
    with open('models/saved_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    logging.info("Model and scaler have been saved.")


if __name__ == '__main__':
    # Replace the path with your actual CSV file location.
    data_filepath = '/Users/krishlakhani/PycharmProjects/ML_Engineer_Assignment/data/MLE-Assignment.csv'

    if not os.path.exists(data_filepath):
        logging.error("Data file not found at %s", data_filepath)
    else:
        run_pipeline(data_filepath)
