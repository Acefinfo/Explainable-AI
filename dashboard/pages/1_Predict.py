import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Adds the parent director to path for imports to work in subspages
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.loader import (
    load_model, load_scaler, load_train_data,
    get_feature_names, MODEL_OPTIONS
)

st.set_page_config(page_title="Predict", page_icon="", layout="wide")
st.title("Student Pass/Fail Prediction")

# --------------------------------------------------
# Load resources
# --------------------------------------------------
best_model_name = "Gradient Boosting (Best)"
best_model      = load_model(MODEL_OPTIONS[best_model_name])
scaler          = load_scaler()
X_train, _      = load_train_data()
feature_names   = get_feature_names()

# --------------------------------------------------
# Sidebar — model switcher 
# --------------------------------------------------
st.sidebar.markdown("### Model selection")
st.sidebar.info("Gradient Boosting is selected by default (best F1: 95.1%)")
selected_model_name = st.sidebar.selectbox(
    "Switch to a different model:",
    list(MODEL_OPTIONS.keys()),
    index=0
)
model = load_model(MODEL_OPTIONS[selected_model_name])

if selected_model_name == best_model_name:
    st.sidebar.success("Using best model")
else:
    st.sidebar.warning(f"Using {selected_model_name}")

# --------------------------------------------------
# Input form
# --------------------------------------------------
st.subheader("Enter student details")
st.caption("Fill in the student information below. Fields marked with * are the most influential.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Academic**")

    G1 = st.number_input(
        "G1 — first period grade * (0-20)",
        min_value=0, max_value=20, value=10, step=1
    )
    G2 = st.number_input(
        "G2 — second period grade * (0-20)",
        min_value=0, max_value=20, value=11, step=1
    )
    studytime = st.selectbox(
        "Study time per week *",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "less than 2 hours",
            2: "2 to 5 hours",
            3: "5 to 10 hours",
            4: "More than 10 hours"
        }[x],
        index=1
    )
    failures = st.selectbox(
        "Past class failures *",
        options=[0, 1, 2, 3],
        format_func=lambda x: f"{x} failure{'s' if x != 1 else ''}",
        index=0
    )

with col2:
    st.markdown("**Personal**")

    age = st.number_input(
        "Age (15-22)",
        min_value=15, max_value=22, value=17, step=1
    )
    absences = st.number_input(
        "Number of absences * (0-93)",
        min_value=0, max_value=93, value=4, step=1
    )
    goout = st.selectbox(
        "Going out with friends *",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: {
            1: "Rarely out",
            2: "Occasionally out",
            3: "Sometimes out",
            4: "Frequent out",
            5: "Always out"
        }[x],
        index=2
    )
    health = st.selectbox(
        "Current health status",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: {
            1: "Very poor health",
            2: "Poor health",
            3: "Fair health",
            4: "Good health",
            5: "Excellent health"
        }[x],
        index=2
    )
    romantic = st.selectbox("In a romantic relationship?", ["No", "Yes"])

with col3:
    st.markdown("**Family & School**")

    Medu = st.selectbox(
        "Mother's education level",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: {
            0: "None",
            1: "Primary (up to 4th grade)",
            2: "5th to 9th grade",
            3: "Secondary education",
            4: "Higher education"
        }[x],
        index=2
    )
    Fedu = st.selectbox(
        "Father's education level",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: {
            0: "None",
            1: "Primary (up to 4th grade)",
            2: "5th to 9th grade",
            3: "Secondary education",
            4: "Higher education"
        }[x],
        index=2
    )
    famrel = st.slider("Family relationship quality (1-5)", 1, 5, 4)
    schoolsup = st.selectbox("Extra school support?", ["No", "Yes"])
    higher    = st.selectbox("Wants higher education?", ["Yes", "No"])
    internet  = st.selectbox("Internet access at home?", ["Yes", "No"])

# --------------------------------------------------
# Build input row matching training feature order
# --------------------------------------------------
numerical = [
    "age","Medu","Fedu","traveltime","studytime","failures",
    "famrel","freetime","goout","Dalc","Walc","health",
    "absences","G1","G2"
]

# Creates a row matching training features, then applies scaling to numerical features
def build_input(feature_names):
    row = {f: 0 for f in feature_names}
    row["G1"]        = G1
    row["G2"]        = G2
    row["studytime"] = studytime
    row["failures"]  = failures
    row["age"]       = age
    row["absences"]  = absences
    row["goout"]     = goout
    row["health"]    = health
    row["Medu"]      = Medu
    row["Fedu"]      = Fedu
    row["famrel"]    = famrel
    row["romantic_yes"]  = 1 if romantic == "Yes" else 0
    row["schoolsup_yes"] = 1 if schoolsup == "Yes" else 0
    row["higher_yes"]    = 1 if higher == "Yes" else 0
    row["internet_yes"]  = 1 if internet == "Yes" else 0

    df = pd.DataFrame([row])[feature_names]
    df[numerical] = scaler.transform(df[numerical])
    return df

# --------------------------------------------------
# Predict button
# --------------------------------------------------
st.markdown("---")
predict_btn = st.button("Run Prediction", type="primary", use_container_width=True)

if predict_btn:
    input_df   = build_input(feature_names)

    st.session_state["input_df"]    = input_df
    st.session_state["model_name"]  = selected_model_name

    prediction = model.predict(input_df)[0]
    proba      = model.predict_proba(input_df)[0]
    pass_prob  = round(proba[1] * 100, 1)
    fail_prob  = round(proba[0] * 100, 1)

    st.markdown("---")
    res_col, shap_col = st.columns([1, 2])

    with res_col:
        if prediction == 1:
            st.success("### PASS")
            st.metric("Pass probability", f"{pass_prob}%")
            st.metric("Fail probability", f"{fail_prob}%")
        else:
            st.error("### FAIL")
            st.metric("Fail probability", f"{fail_prob}%")
            st.metric("Pass probability", f"{pass_prob}%")

        st.caption(f"Model used: **{selected_model_name}**")

        if selected_model_name != best_model_name:
            best_pred  = best_model.predict(input_df)[0]
            best_proba = round(best_model.predict_proba(input_df)[0][1] * 100, 1)
            st.info(
                f"Best model (Gradient Boosting) predicts: "
                f"**{'PASS' if best_pred == 1 else 'FAIL'}** "
                f"({best_proba}% pass confidence)"
            )

    with shap_col:
        st.markdown("**Top features driving this prediction (SHAP)**")
        try:
            if selected_model_name in ["Gradient Boosting (Best)",
                                        "Random Forest", "Decision Tree"]:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(input_df)
                if isinstance(shap_vals, list):
                    sv = shap_vals[1][0]
                elif shap_vals.ndim == 3:
                    sv = shap_vals[0, :, 1]
                else:
                    sv = shap_vals[0]
            else:
                explainer = shap.Explainer(model, X_train)
                shap_vals = explainer(input_df)
                sv = shap_vals.values[0]
                if sv.ndim == 2:
                    sv = sv[:, 1]

            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP":    sv
            }).reindex(pd.Series(sv).abs().sort_values(ascending=False).index)
            shap_df = shap_df.head(10).sort_values("SHAP")

            colors = ["#ef4444" if v < 0 else "#3b82f6" for v in shap_df["SHAP"]]

            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")
            ax.barh(shap_df["Feature"], shap_df["SHAP"], color=colors)
            ax.axvline(0, color="white", linewidth=0.8)
            ax.set_xlabel("SHAP value", color="white")
            ax.set_title(
                "Blue = pushes toward Pass   |   Red = pushes toward Fail",
                color="white", fontsize=9
            )
            ax.tick_params(colors="white", labelsize=9)
            ax.spines[:].set_color("#333")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")