import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.loader import load_cv_results, load_spearman_results

st.set_page_config(page_title="About", page_icon="", layout="wide")
st.title("About This Project")
st.caption("BSc (Hons) Computing 2025/26 · Aashutosh Thapa · c7466915 · Leeds Beckett University")

st.markdown("---")

# --------------------------------------------------
# Project overview
# --------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Project aim")
    st.markdown(
        """
        This system was built as part of a Level 6 Production Project to design,
        implement and evaluate an **Explainable AI (XAI)** system that provides
        accurate predictions while offering clear and comprehensive explanations
        for every decision it makes.

        The project integrates two leading explainability techniques — **SHAP**
        and **LIME** — within a single machine learning pipeline, and presents
        the results through this interactive dashboard to enhance interpretability,
        trust and informed decision-making.
        """
    )

with col2:
    st.subheader("Research gap addressed")
    st.markdown(
        """
        Existing research highlights that SHAP and LIME are commonly evaluated
        in isolation rather than comparatively *(Doshi-Velez & Kim, 2017;
        Arrieta et al., 2020)*.

        This project addresses that gap by integrating and comparing both methods
        within a single system — evaluating explanation clarity, consistency and
        usability alongside predictive accuracy, in line with ethical AI principles
        outlined by the European Commission (2019).
        """
    )

st.markdown("---")

# --------------------------------------------------
# Dataset and task
# --------------------------------------------------
st.subheader("Dataset & task")

d1, d2, d3, d4 = st.columns(4)
d1.metric("Dataset", "UCI Student Performance")
d2.metric("Total students", "395")
d3.metric("Features", "41 (after encoding)")
d4.metric("Task", "Binary classification")

st.markdown(
    """
    The **UCI Student Performance Dataset** contains demographic, social and
    academic information about secondary school students in Portugal.
    The target variable is whether a student **passes** (final grade G3 ≥ 10)
    or **fails**. G1 and G2 (first and second period grades) were retained as
    features — G3 was dropped to prevent data leakage.
    """
)

st.markdown("---")

# --------------------------------------------------
# Models
# --------------------------------------------------
st.subheader("Models trained")

m1, m2, m3, m4, m5 = st.columns(5)
for col, (name, detail) in zip(
    [m1, m2, m3, m4, m5],
    [
        ("Logistic Regression", "Baseline linear model · max_iter=1000"),
        ("Decision Tree",       "Baseline tree · random_state=42"),
        ("Random Forest",       "100 estimators · random_state=42"),
        ("Gradient Boosting",   "Best model · F1: 95.1%"),
        ("SVM",                 "RBF kernel · probability=True"),
    ]
):
    col.markdown(
        f"""<div style="background:#1a1a1a;border:1px solid #333;border-radius:8px;
                        padding:12px;text-align:center;min-height:90px;">
                <div style="font-size:12px;font-weight:600;color:#e2e8f0;
                            margin-bottom:6px;">{name}</div>
                <div style="font-size:11px;color:#64748b;">{detail}</div>
            </div>""",
        unsafe_allow_html=True
    )

st.markdown("---")

# --------------------------------------------------
# XAI methods
# --------------------------------------------------
st.subheader("Explainability methods")

x1, x2 = st.columns(2)

with x1:
    st.markdown(
        """<div style="background:#0f2027;border:1px solid #1d4ed8;border-radius:8px;
                       padding:16px;">
               <div style="font-size:15px;font-weight:600;color:#93c5fd;
                           margin-bottom:8px;">SHAP — SHapley Additive exPlanations</div>
               <div style="font-size:13px;color:#94a3b8;line-height:1.7">
                   Based on cooperative game theory (Lundberg & Lee, 2017).
                   Provides both <strong style="color:#e2e8f0">global</strong> feature
                   importance across all predictions and
                   <strong style="color:#e2e8f0">local</strong> explanations for
                   individual instances.<br><br>
                   Uses TreeExplainer for tree-based models (exact and fast)
                   and LinearExplainer for Logistic Regression.
                   <br><br>
                   <strong style="color:#e2e8f0">Key property:</strong>
                   Consistent and theoretically grounded — same input always
                   produces the same explanation.
               </div>
           </div>""",
        unsafe_allow_html=True
    )

with x2:
    st.markdown(
        """<div style="background:#1a1200;border:1px solid #92400e;border-radius:8px;
                       padding:16px;">
               <div style="font-size:15px;font-weight:600;color:#fcd34d;
                           margin-bottom:8px;">LIME — Local Interpretable Model-Agnostic Explanations</div>
               <div style="font-size:13px;color:#94a3b8;line-height:1.7">
                   Explains individual predictions by approximating the model
                   locally with a simpler interpretable model
                   (Ribeiro, Singh & Guestrin, 2016).<br><br>
                   Applied using LimeTabularExplainer trained on the full
                   training dataset. Explains 5 test instances per model
                   with num_features=10.
                   <br><br>
                   <strong style="color:#e2e8f0">Key property:</strong>
                   Fast and model-agnostic — but non-deterministic.
                   Different random seeds produce slightly different explanations.
               </div>
           </div>""",
        unsafe_allow_html=True
    )

st.markdown("---")

# --------------------------------------------------
# Key findings
# --------------------------------------------------
st.subheader("Key findings")

f1, f2, f3 = st.columns(3)

with f1:
    st.markdown(
        """<div style="background:#0f1f0f;border:1px solid #166534;
                       border-radius:8px;padding:14px;">
               <div style="font-size:13px;font-weight:600;color:#86efac;
                           margin-bottom:6px;">Best model</div>
               <div style="font-size:12px;color:#94a3b8;line-height:1.6">
                   Gradient Boosting achieved the highest F1 of <strong
                   style="color:#e2e8f0">95.1%</strong> in 5-fold CV.
                   Logistic Regression was surprisingly competitive, suggesting
                   the data is largely linearly separable. SVM performed
                   weakest despite its complexity.
               </div>
           </div>""",
        unsafe_allow_html=True
    )

with f2:
    st.markdown(
        """<div style="background:#0f1520;border:1px solid #1d4ed8;
                       border-radius:8px;padding:14px;">
               <div style="font-size:13px;font-weight:600;color:#93c5fd;
                           margin-bottom:6px;">Most important features</div>
               <div style="font-size:12px;color:#94a3b8;line-height:1.6">
                   Both SHAP and LIME consistently identify <strong
                   style="color:#e2e8f0">G2</strong> (second period grade)
                   and <strong style="color:#e2e8f0">G1</strong> (first period
                   grade) as by far the strongest predictors — confirming that
                   prior academic performance dominates over social and
                   demographic factors.
               </div>
           </div>""",
        unsafe_allow_html=True
    )

with f3:
    st.markdown(
        """<div style="background:#1a1000;border:1px solid #92400e;
                       border-radius:8px;padding:14px;">
               <div style="font-size:13px;font-weight:600;color:#fcd34d;
                           margin-bottom:6px;">SHAP vs LIME agreement</div>
               <div style="font-size:12px;color:#94a3b8;line-height:1.6">
                   Mean Spearman correlation of <strong
                   style="color:#e2e8f0">0.477</strong> across 5 test samples
                   (all p &lt; 0.01). Both methods agree on top features but
                   diverge on lower-ranked ones — consistent with LIME's known
                   instability on less important features.
               </div>
           </div>""",
        unsafe_allow_html=True
    )

st.markdown("---")

# --------------------------------------------------
# CV results summary
# --------------------------------------------------
st.subheader("Model performance summary (5-fold cross-validation)")

try:
    cv_df = load_cv_results()
    st.dataframe(cv_df, use_container_width=True, hide_index=True)
except Exception:
    st.info("CV results not found — run notebook 03 to generate them.")

st.markdown("---")

# --------------------------------------------------
# Methodology
# --------------------------------------------------
st.subheader("Methodology")

st.markdown(
    """
    This project followed an **Agile iterative development** methodology,
    chosen because the project involves evolving requirements such as model
    tuning and iterative evaluation of explainability techniques.

    Development was structured across 7 Jupyter notebooks:
    """
)

notebooks = [
    ("01", "Data exploration",        "EDA, target variable creation, class distribution analysis"),
    ("02", "Preprocessing",           "Encoding, scaling, stratified train/test split, scaler saved"),
    ("03", "Model training",          "5 models trained, saved as .pkl, 5-fold CV added"),
    ("04", "Model evaluation",        "Accuracy, F1, ROC-AUC, confusion matrices, PR curves"),
    ("05", "SHAP explanations",       "Global + local SHAP for all 5 models, plots saved"),
    ("06", "LIME explanations",       "25 HTML explanations generated (5 samples × 5 models)"),
    ("07", "SHAP vs LIME comparison", "Spearman correlation, LIME stability test, rank comparison"),
]

for num, name, desc in notebooks:
    st.markdown(
        f"""<div style="display:flex;align-items:flex-start;gap:12px;
                        padding:8px 0;border-bottom:1px solid #1e293b;">
                <div style="background:#1e293b;color:#94a3b8;border-radius:5px;
                            padding:3px 8px;font-size:11px;font-family:monospace;
                            flex-shrink:0;margin-top:1px;">0{num}</div>
                <div>
                    <div style="font-size:13px;font-weight:500;color:#e2e8f0;">
                        {name}</div>
                    <div style="font-size:12px;color:#64748b;">{desc}</div>
                </div>
            </div>""",
        unsafe_allow_html=True
    )

st.markdown("---")


st.markdown("---")

# --------------------------------------------------
# Tech stack
# --------------------------------------------------
st.subheader("Tech stack")

tech = [
    ("Python", "3.10.11"),
    ("scikit-learn", "ML models + evaluation"),
    ("SHAP", "Global + local explanations"),
    ("LIME", "Instance-level explanations"),
    ("Streamlit", "Interactive dashboard"),
    ("pandas / numpy", "Data processing"),
    ("matplotlib", "Visualisations"),
    ("scipy", "Spearman correlation"),
    ("joblib", "Model persistence"),
]

tcols = st.columns(3)
for i, (lib, desc) in enumerate(tech):
    with tcols[i % 3]:
        st.markdown(
            f"""<div style="background:#1a1a1a;border:1px solid #333;
                            border-radius:6px;padding:8px 12px;margin-bottom:8px;
                            display:flex;justify-content:space-between;
                            align-items:center;">
                    <span style="font-size:12px;font-weight:500;color:#e2e8f0;">
                        {lib}</span>
                    <span style="font-size:11px;color:#64748b;">{desc}</span>
                </div>""",
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption(
    "BSc (Hons) Computing 2025/26 · Aashutosh Thapa · c7466915 · "
    "Leeds Beckett University · Supervisor: Saroj Sharma,Rohit Raj Pandey"
)