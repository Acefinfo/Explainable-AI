
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)


# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models"
EVAL_PATH = BASE_DIR / "evaluation"

sns.set_style("whitegrid")


# -----------------------------
# Utility Functions
# -----------------------------

def load_test_data():
    X_test = pd.read_csv(DATA_PATH / "X_test.csv")
    y_test = pd.read_csv(DATA_PATH / "y_test.csv").values.ravel()
    return X_test, y_test


def load_models():
    models = {
        "Logistic Regression": joblib.load(MODEL_PATH / "logistic_regression.pkl"),
        "Decision Tree": joblib.load(MODEL_PATH / "decision_tree.pkl"),
        "Random Forest": joblib.load(MODEL_PATH / "random_forest.pkl"),
        "Gradient Boosting": joblib.load(MODEL_PATH / "gradient_boosting.pkl"),
        "Support Vector Machine": joblib.load(MODEL_PATH / "support_vector_machine.pkl"),
    }
    return models


def get_model_probabilities(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min())
    else:
        return None


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = get_model_probabilities(model, X_test)

    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
    }

    return results, y_pred, y_prob


# -----------------------------
# Evaluation Functions
# -----------------------------

def save_confusion_matrix(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_path = EVAL_PATH / "confusion_matrices"
    cm_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(cm_path / f"{name.replace(' ', '_')}_cm.png")
    plt.close()


def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        y_prob = get_model_probabilities(model, X_test)
        if y_prob is None:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()

    plt.savefig(EVAL_PATH / "roc_curve_comparison.png")
    plt.close()


def plot_pr_curves(models, X_test, y_test):
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        y_prob = get_model_probabilities(model, X_test)
        if y_prob is None:
            continue

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend()

    plt.savefig(EVAL_PATH / "precision_recall_curve_comparison.png")
    plt.close()


# -----------------------------
# Main Function
# -----------------------------

def main():
    print("Starting model evaluation...")

    EVAL_PATH.mkdir(exist_ok=True)

    X_test, y_test = load_test_data()
    models = load_models()

    results = []

    for name, model in models.items():
        print(f"Evaluating {name}...")

        metrics, y_pred, y_prob = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)

        save_confusion_matrix(name, y_test, y_pred)

        print(classification_report(y_test, y_pred))

    results_df = pd.DataFrame(results)
    results_df.sort_values("F1-Score", ascending=False, inplace=True)

    results_df.to_csv(EVAL_PATH / "performance_metrics.csv", index=False)

    plot_roc_curves(models, X_test, y_test)
    plot_pr_curves(models, X_test, y_test)

    print("\nEvaluation completed successfully.")
    print("\nFinal Model Ranking:")
    print(results_df)


if __name__ == "__main__":
    main()