import pandas as pd 
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_data(data_dir: Path):
    """
    Load preprocessed training and testing datasets.

    Returns:
    X_train, X_test, y_train, y_test
    """
    x_train = pd.read_csv(data_dir/"X_train.csv")
    x_test = pd.read_csv(data_dir/"X_test.csv")

    y_train = pd.read_csv(data_dir/"y_train.csv").squeeze()
    y_test = pd.read_csv(data_dir/"y_test.csv").squeeze()

    return x_train, x_test, y_train, y_test

def model_initialization():
    """
    Initialize and return all models used in the project.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter = 1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Support Vector Machine": SVC(kernel= "rbf", probability=True, random_state=42)
    }
    return models

def train_models(models, x_train, y_train, x_test, y_test):
    """
    Train each model and compute accuracy.

    Returns:
    trained_models, results_df
    """
    trained_models = {}
    results = []

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        y_prediction = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_prediction)

        trained_models[model_name] = model
        results.append({
            "Model": model_name,
            "Accuracy": accuracy
        })

        print (f"{model_name} trained successfully || Accuracy: {accuracy:.4f}")

    result = pd.DataFrame(results)
    return trained_models, result

def save_trained_models(trained_models, model_dir: Path):
    """
    Save trained models as .pkl files.
    """
    model_dir.mkdir(parents= True, exist_ok=True)

    for model_name,model in trained_models.items():
        filename = model_name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, model_dir/filename)

    print("All models trained and saved successfully")

def main():

    # Resolve project root directory
    BASE_DIR = Path(__file__).resolve().parent.parent

    DATA_DIR = BASE_DIR/"data"/"processed"
    MODEL_DIR = BASE_DIR/"models"

    # Load preprocessed data
    x_train, x_test, y_train, y_test = load_data(DATA_DIR)

    # Initialize models
    models = model_initialization()

    # Train and evulate all the models
    trained_models, result = train_models(models, x_train, y_train, x_test, y_test)

    # Save the trained models
    save_trained_models(trained_models, MODEL_DIR)

    # Display training summary
    print("\nTRAINING SUMMARY")
    print("-" * 50)
    print(result.sort_values(by="Accuracy", ascending=False))
    print("\nModel training pipeline completed successfully.")

# Entry point for the script
if __name__ == "__main__":
    main()