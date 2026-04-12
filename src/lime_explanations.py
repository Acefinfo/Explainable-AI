from pathlib import Path
import pandas as pd
import joblib
import os

from lime.lime_tabular import LimeTabularExplainer


# ==============================
# PATH SETUP
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models"
OUTPUT_PATH = BASE_DIR / "evaluation" / "lime_explanations"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ==============================
# LOAD DATA
# ==============================

def load_data():
    X_train = pd.read_csv(DATA_PATH / "X_train.csv")
    X_test = pd.read_csv(DATA_PATH / "X_test.csv")
    return X_train, X_test


# ==============================
# LOAD MODELS
# ==============================

def load_models():
    models = {
        "logistic_regression": joblib.load(MODEL_PATH / "logistic_regression.pkl"),
        "decision_tree": joblib.load(MODEL_PATH / "decision_tree.pkl"),
        "random_forest": joblib.load(MODEL_PATH / "random_forest.pkl"),
        "gradient_boosting": joblib.load(MODEL_PATH / "gradient_boosting.pkl"),
        "svm": joblib.load(MODEL_PATH / "support_vector_machine.pkl")
    }
    return models


# ==============================
# CREATE LIME EXPLAINER
# ==============================

def create_explainer(X_train):
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["Fail", "Pass"],   
        mode="classification"
    )
    return explainer


# ==============================
# GENERATE EXPLANATIONS
# ==============================

def generate_explanations(num_samples=5, num_features=10):
    print("Generating LIME explanations...\n")

    X_train, X_test = load_data()
    models = load_models()
    explainer = create_explainer(X_train)

    for i in range(num_samples):
        print(f"Processing Sample {i}...")

        for name, model in models.items():

            try:
                exp = explainer.explain_instance(
                    X_test.iloc[i].values,
                    model.predict_proba,
                    num_features=num_features
                )

                file_name = f"lime_{name}_sample_{i}.html"
                file_path = OUTPUT_PATH / file_name

                exp.save_to_file(str(file_path))

                print(f" Saved: {file_name}")

            except Exception as e:
                print(f" Error in {name} for sample {i}: {e}")

    print("\n LIME explanation generation completed!")


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    generate_explanations(num_samples=5, num_features=10)