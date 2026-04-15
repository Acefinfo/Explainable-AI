import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# ---------------------------------------------------
# Resolve paths relative to this file's location
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_DIR = BASE_DIR / "models"
DATA_DIR  = BASE_DIR / "data" / "processed"
EVAL_DIR  = BASE_DIR / "evaluation"

# ---------------------------------------------------
# Model definitions
# ---------------------------------------------------
MODEL_OPTIONS = {
    "Gradient Boosting (Best)": "gradient_boosting.pkl",
    "Random Forest":            "random_forest.pkl",
    "Logistic Regression":      "logistic_regression.pkl",
    "Decision Tree":            "decision_tree.pkl",
    "Support Vector Machine":   "support_vector_machine.pkl",
}

@st.cache_resource
def load_model(model_filename: str):
    return joblib.load(MODEL_DIR / model_filename)

@st.cache_resource
def load_scaler():
    return joblib.load(MODEL_DIR / "scaler.pkl")

@st.cache_data
def load_train_data():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
    return X_train, y_train

@st.cache_data
def load_test_data():
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()
    return X_test, y_test

@st.cache_data
def load_cv_results():
    return pd.read_csv(EVAL_DIR / "cv_results.csv")

@st.cache_data
def load_spearman_results():
    return pd.read_csv(EVAL_DIR / "shap_lime_spearman.csv")

def get_feature_names():
    X_train, _ = load_train_data()
    return X_train.columns.tolist()