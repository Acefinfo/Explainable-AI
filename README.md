# Explainable AI (XAI) Project

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Streamlit-1.28+-red.svg" alt="Streamlit Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-orange.svg" alt="Status">
  <a href="https://aiexplainable.streamlit.app/"><img src="https://img.shields.io/badge/Hosted-Streamlit%20Cloud-purple.svg" alt="Hosted URL"></a>
</p>

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Features](#-features)
3. [Project Structure](#-project-structure)
4. [Technology Stack](#-technology-stack)
5. [Installation](#-installation)
6. [Usage](#-usage)
7. [Notebooks](#-notebooks)
8. [Models](#-models)
9. [Evaluation Metrics](#-evaluation-metrics)
10. [Explanation Methods](#-explanation-methods)
11. [Dashboard](#-dashboard)
12. [Results](#-results)
13. [Contributing](#-contributing)
14. [License](#-license)

---

## 📖 Project Overview

This project implements a comprehensive **Explainable AI (XAI)** system for student performance prediction. The main goal is to not only build accurate machine learning models but also to provide transparent and interpretable explanations for their predictions.

The system predicts whether a student will pass or fail based on various demographic, social, and academic features. It compares multiple machine learning algorithms and evaluates their explanations using two state-of-the-art XAI methods: **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)**.

### 🎯 Objectives

- **Build** accurate predictive models for student performance
- **Explain** model predictions using interpretable AI techniques
- **Compare** different explanation methods (SHAP vs LIME)
- **Visualize** feature importance and model behavior
- **Deploy** an interactive dashboard for predictions and explanations

---

## ✨ Features

### Core Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Support** | Supports 5 different ML algorithms: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and SVM |
| **SHAP Explanations** | Provides SHAP-based feature importance and local explanations |
| **LIME Explanations** | Generates LIME-based local explanations with HTML visualization |
| **Model Comparison** | Compares models across multiple evaluation metrics |
| **Interactive Dashboard** | Streamlit-based web interface for predictions and explanations |
| **Cross-Validation** | Performs k-fold cross-validation for robust model evaluation |
| **Explanation Stability** | Tests stability of LIME explanations across multiple runs |

### Technical Features

- **Data Preprocessing Pipeline**: Automated data loading, encoding, and scaling
- **Model Persistence**: Save/load trained models using joblib
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC, MCC
- **Visualization**: Confusion matrices, ROC curves, Precision-Recall curves
- **Statistical Comparison**: Spearman correlation between SHAP and LIME importance

---

## 📂 Project Structure

```
Explainable AI/
├── .devcontainer/              # Development container configuration
├── .git/                        # Git repository
├── .gitignore                   # Git ignore rules
├── dashboard/                   # Streamlit dashboard
│   ├── Home.py                  # Main dashboard page
│   ├── pages/                   # Dashboard pages
│   │   ├── 1_Predict.py         # Prediction page
│   │   ├── 2_Explanations.py   # Explanations page
│   │   ├── 3_Model_Comparison.py # Model comparison page
│   │   └── 4_About.py           # About page
│   └── utils/
│       └── loader.py            # Utility functions for dashboard
├── data/                        # Data directory
│   ├── processed/               # Preprocessed data
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   └── raw/                     # Raw data
├── evaluation/                  # Evaluation results
│   ├── confusion_matrices/      # Confusion matrix plots
│   ├── lime_explanations/       # LIME HTML explanations
│   ├── shap_comparison/         # SHAP plots
│   ├── cv_results.csv           # Cross-validation results
│   ├── performance_metrics.csv  # Model performance metrics
│   ├── shap_lime_comparison_sample0.csv
│   ├── shap_lime_spearman.csv   # SHAP-LIME correlation
│   ├── lime_stability_test.png
│   ├── precision_recall_curve_comparison.png
│   └── roc_curve_comparison.png
├── models/                      # Trained model files
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── gradient_boosting.pkl
│   └── support_vector_machine.pkl
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   ├── 05_shap_explanations.ipynb
│   ├── 06_lime_explanations.ipynb
│   └── 07_shap_vs_lime_comparison.ipynb
├── src/                         # Source code
│   ├── data_preprocessing.py    # Data preprocessing functions
│   ├── train_models.py          # Model training script
│   ├── evaluate_models.py       # Model evaluation script
│   ├── shap_explanations.py     # SHAP explanation generation
│   └── lime_explanations.py     # LIME explanation generation
├── requirements.txt             # Python dependencies
└── venv/                        # Virtual environment
```

---

## 🛠 Technology Stack

### Programming Language

- **Python 3.10**

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | latest | Data manipulation and analysis |
| `numpy` | latest | Numerical computing |
| `scikit-learn` | latest | Machine learning algorithms |
| `matplotlib` | latest | Data visualization |
| `seaborn` | latest | Statistical graphics |
| `shap` | latest | SHAP explanations |
| `lime` | latest | LIME explanations |
| `xgboost` | latest | Gradient boosting |
| `joblib` | latest | Model serialization |
| `streamlit` | latest | Web dashboard |

### Development Tools

- **Jupyter Notebook**: Interactive computing
- **Git**: Version control
- **VS Code**: Code editor

---

## 🚀 Installation

### Prerequisites

- Python 3.10
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd "Explainable AI"
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv

# Activate virtual environment
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (CMD)
venv\Scripts\activate.bat

# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap lime xgboost joblib streamlit
```

### Step 4: Verify Installation

```bash
python -c "import streamlit; import shap; import lime; print('All packages installed successfully!')"
```

---

## 💻 Usage

### Option 1: Run the Dashboard (Local)

The easiest way to explore the project is through the interactive Streamlit dashboard.

```bash
# Navigate to project directory
cd "Explainable AI"

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run the dashboard
streamlit run dashboard/Home.py
```

The dashboard will open in your browser at `http://localhost:8500`.

### Option 2: Try Online (Streamlit Cloud)

You can also access the deployed dashboard online without installing anything:

🔗 **Live Dashboard**: https://aiexplainable.streamlit.app/

### Option 2: Run Individual Scripts

#### Data Preprocessing

```bash
python src/data_preprocessing.py
```

#### Model Training

```bash
python src/train_models.py
```

#### Model Evaluation

```bash
python src/evaluate_models.py
```

#### Generate SHAP Explanations

```bash
python src/shap_explanations.py
```

#### Generate LIME Explanations

```bash
python src/lime_explanations.py
```

### Option 3: Use Jupyter Notebooks

Open any notebook in the `notebooks/` directory:

```bash
jupyter notebook
```

Then navigate to:
- `notebooks/01_data_exploration.ipynb` - Explore the dataset
- `notebooks/02_preprocessing.ipynb` - Preprocess data
- `notebooks/03_model_training.ipynb` - Train models
- `notebooks/04_model_evaluation.ipynb` - Evaluate models
- `notebooks/05_shap_explanations.ipynb` - Generate SHAP explanations
- `notebooks/06_lime_explanations.ipynb` - Generate LIME explanations
- `notebooks/07_shap_vs_lime_comparison.ipynb` - Compare SHAP and LIME

---

## 📓 Notebooks

### Notebook Descriptions

| Notebook | Description |
|----------|-------------|
| **01_data_exploration.ipynb** | Initial data exploration, understanding features, basic statistics |
| **02_preprocessing.ipynb** | Data cleaning, feature engineering, encoding, scaling |
| **03_model_training.ipynb** | Training 5 different ML models with hyperparameter tuning |
| **04_model_evaluation.ipynb** | Comprehensive model evaluation using various metrics |
| **05_shap_explanations.ipynb** | Generating SHAP values and visualizations for all models |
| **06_lime_explanations.ipynb** | Generating LIME explanations and HTML reports |
| **07_shap_vs_lime_comparison.ipynb** | Comparing SHAP and LIME explanation methods |

---

## 🤖 Models

The project implements and compares five machine learning models:

### 1. Logistic Regression
- **Type**: Linear classifier
- **Algorithm**: Maximum likelihood estimation
- **Use Case**: Baseline model, interpretable

### 2. Decision Tree
- **Type**: Tree-based classifier
- **Algorithm**: Recursive partitioning
- **Use Case**: Interpretable rules

### 3. Random Forest
- **Type**: Ensemble (bagging)
- **Algorithm**: Multiple decision trees
- **Use Case**: High accuracy, robust

### 4. Gradient Boosting
- **Type**: Ensemble (boosting)
- **Algorithm**: Sequential error correction
- **Use Case**: Best accuracy, complex patterns

### 5. Support Vector Machine (SVM)
- **Type**: Kernel-based classifier
- **Algorithm**: Maximum margin hyperplane
- **Use Case**: Non-linear decision boundaries

---

## 📊 Evaluation Metrics

The project evaluates models using multiple metrics:

| Metric | Description | Range |
|--------|-------------|-------|
| **Accuracy** | Correct predictions / Total predictions | [0, 1] |
| **Precision** | True Positives / (True Positives + False Positives) | [0, 1] |
| **Recall** | True Positives / (True Positives + False Negatives) | [0, 1] |
| **F1-Score** | Harmonic mean of Precision and Recall | [0, 1] |
| **ROC-AUC** | Area under ROC curve | [0, 1] |
| **MCC** | Matthews Correlation Coefficient | [-1, 1] |
| **Balanced Accuracy** | Accuracy accounting for class imbalance | [0, 1] |

### Evaluation Outputs

- **Confusion Matrices**: Visual representation of predictions
- **ROC Curves**: True Positive Rate vs False Positive Rate
- **Precision-Recall Curves**: Precision vs Recall at different thresholds
- **Cross-Validation Results**: K-fold CV scores for each model

---

## 🔍 Explanation Methods

### SHAP (SHapley Additive exPlanations)

SHAP provides consistent and locally accurate feature attributions based on game theory concepts.

#### Key Features:
- **Global Importance**: Feature importance across all predictions
- **Local Importance**: Explanation for individual predictions
- **Dependence Plots**: Show feature interactions
- **Waterfall Plots**: Visual breakdown of prediction

#### SHAP Outputs:
- `shap_summary_rf.png` - Random Forest summary plot
- `shap_summary_gb.png` - Gradient Boosting summary plot
- `shap_bar_rf.png` - Feature importance bar chart
- `shap_waterfall_logistic.png` - Individual prediction breakdown

### LIME (Local Interpretable Model-agnostic Explanations)

LIME explains individual predictions by approximating model behavior locally.

#### Key Features:
- **Local Surrogate Model**: Approximates model locally
- **Feature Perturbation**: Tests feature importance
- **HTML Output**: Interactive explanation reports
- **Stability Testing**: Evaluates explanation consistency

#### LIME Outputs:
- `lime_{model}_sample_{i}.html` - HTML explanation files
- `lime_stability_test.png` - Stability analysis plot

### SHAP vs LIME Comparison

| Aspect | SHAP | LIME |
|--------|------|------|
| **Theoretical Basis** | Game theory (Shapley values) | Local linear approximation |
| **Global Explanations** | Yes | No (local only) |
| **Consistency** | Guaranteed | Not guaranteed |
| **Speed** | Faster for tree models | Slower, requires sampling |
| **Output Format** | Plots, values | HTML, text |

---

## 📱 Dashboard

The Streamlit dashboard provides an interactive interface with 4 main pages. You can run it locally or access it online:

### 🚀 Live Demo

**Try it now**: https://aiexplainable.streamlit.app/

The dashboard is hosted on Streamlit Cloud and includes all features below.

### 📄 Local Pages

### 1. Home Page (`dashboard/Home.py`)
- Project introduction
- Quick navigation to other pages
- Key metrics overview

### 2. Predict Page (`dashboard/pages/1_Predict.py`)
- Input student features
- Get instant prediction
- View prediction probability

### 3. Explanations Page (`dashboard/pages/2_Explanations.py`)
- Select model and sample
- View SHAP explanations
- View LIME explanations
- Compare both methods

### 4. Model Comparison Page (`dashboard/pages/3_Model_Comparison.py`)
- Compare all models
- View performance metrics
- Visualize ROC curves
- Feature importance comparison

### 5. About Page (`dashboard/pages/4_About.py`)
- Project documentation
- Methodology explanation
- References and links

---

## 📈 Results

### Model Performance (Sample)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.75 | ~0.76 | ~0.74 | ~0.75 | ~0.82 |
| Decision Tree | ~0.70 | ~0.71 | ~0.69 | ~0.70 | ~0.75 |
| Random Forest | ~0.80 | ~0.81 | ~0.79 | ~0.80 | ~0.87 |
| Gradient Boosting | ~0.82 | ~0.83 | ~0.81 | ~0.82 | ~0.89 |
| SVM | ~0.78 | ~0.79 | ~0.77 | ~0.78 | ~0.85 |

> **Note**: Actual values may vary. Run evaluation to get precise metrics.

### Key Findings

1. **Gradient Boosting** typically achieves the highest accuracy
2. **Random Forest** provides excellent accuracy with good interpretability
3. **SHAP** and **LIME** show high correlation for tree-based models
4. **Feature Importance** varies significantly between models

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guide
- Add docstrings to new functions
- Update documentation for any changes
- Test your code before submitting

---

## 📄 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **SHAP**: Scott Lundberg et al. for the SHAP library
- **LIME**: Marco Tulio Ribeiro et al. for the LIME library
- **Streamlit**: For the amazing dashboard framework
- **Scikit-learn**: For machine learning tools

---

## 📚 References

1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NIPS*.
2. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *KDD*.
3. Student Performance Dataset: UCI Machine Learning Repository

---

## 🔗 Links

- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Documentation](https://lime-ml.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---
