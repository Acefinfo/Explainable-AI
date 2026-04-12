from pathlib import Path
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import os

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models"
EVAL_PATH = BASE_DIR / "evaluation" / "shap_comparison"

# -----------------------------
# Load Data
# -----------------------------
def load_data():
    X_test = pd.read_csv(DATA_PATH / "X_test.csv")
    y_test = pd.read_csv(DATA_PATH / "y_test.csv").values.ravel()
    return X_test, y_test

# -----------------------------
# Load Models
# -----------------------------
def load_models():
    models = {
        "logistic": joblib.load(MODEL_PATH / "logistic_regression.pkl"),
        "decision_tree": joblib.load(MODEL_PATH / "decision_tree.pkl"),
        "random_forest": joblib.load(MODEL_PATH / "random_forest.pkl"),
        "gradient_boosting": joblib.load(MODEL_PATH / "gradient_boosting.pkl"),
        "svm": joblib.load(MODEL_PATH / "support_vector_machine.pkl"),
    }
    return models

# -----------------------------
# Create Explainers
# -----------------------------
def create_explainers(models, X_test):
    explainers = {
        "logistic": shap.Explainer(models["logistic"], X_test),
        "decision_tree": shap.TreeExplainer(models["decision_tree"]),
        "random_forest": shap.TreeExplainer(models["random_forest"]),
        "gradient_boosting": shap.TreeExplainer(models["gradient_boosting"]),
    }

    # SVM (slow → use sample)
    X_sample = X_test.sample(50, random_state=42)
    explainers["svm"] = shap.KernelExplainer(models["svm"].predict, X_sample)

    return explainers, X_sample

# -----------------------------
# Compute SHAP Values
# -----------------------------
def compute_shap_values(explainers, X_test, X_sample):
    shap_values = {
        "logistic": explainers["logistic"](X_test),
        "random_forest": explainers["random_forest"].shap_values(X_test),
        "gradient_boosting": explainers["gradient_boosting"].shap_values(X_test),
        "decision_tree": explainers["decision_tree"].shap_values(X_test),
    }

    # SVM (sample only)
    shap_values["svm"] = explainers["svm"].shap_values(X_sample)

    return shap_values

# -----------------------------
# Save Plots
# -----------------------------
def save_plots(shap_values, X_test):
    os.makedirs(EVAL_PATH, exist_ok=True)

    print("Saving SHAP plots...")

    # Random Forest Summary
    shap.summary_plot(shap_values["random_forest"], X_test, show=False)
    plt.savefig(EVAL_PATH / "shap_summary_rf.png")
    plt.close()

    # Gradient Boosting Summary
    shap.summary_plot(shap_values["gradient_boosting"], X_test, show=False)
    plt.savefig(EVAL_PATH / "shap_summary_gb.png")
    plt.close()

    # Bar Plot (RF)
    shap.summary_plot(
        shap_values["random_forest"], X_test, plot_type="bar", show=False
    )
    plt.savefig(EVAL_PATH / "shap_bar_rf.png")
    plt.close()

    # Waterfall Plot (Logistic)
    shap.plots.waterfall(shap_values["logistic"][0], show=False)
    plt.savefig(EVAL_PATH / "shap_waterfall_logistic.png")
    plt.close()

    print("Plots saved in:", EVAL_PATH)

# -----------------------------
# Main Function
# -----------------------------
def main():
    print("Loading data...")
    X_test, y_test = load_data()

    print("Loading models...")
    models = load_models()

    print("Creating explainers...")
    explainers, X_sample = create_explainers(models, X_test)

    print("Computing SHAP values...")
    shap_values = compute_shap_values(explainers, X_test, X_sample)

    print("Saving plots...")
    save_plots(shap_values, X_test)

    print("SHAP explanation pipeline completed successfully!")

# -----------------------------
# Run Script
# -----------------------------
if __name__ == "__main__":
    main()