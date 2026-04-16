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

if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

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

    # Save to session state for Explanations + Model Comparison pages
    st.session_state["input_df"]   = input_df
    st.session_state["model_name"] = selected_model_name

    prediction = model.predict(input_df)[0]
    proba      = model.predict_proba(input_df)[0]
    pass_prob  = round(proba[1] * 100, 1)
    fail_prob  = round(proba[0] * 100, 1)
    verdict    = "PASS" if prediction == 1 else "FAIL"

    # --------------------------------------------------
    # Save to prediction history
    # --------------------------------------------------
    st.session_state["prediction_history"].append({
        "Model":            selected_model_name,
        "G1":               G1,
        "G2":               G2,
        "Failures":         failures,
        "Absences":         absences,
        "Study time":       studytime,
        "Prediction":       verdict,
        "Pass probability": f"{pass_prob}%",
        "Fail probability": f"{fail_prob}%",
    })

    st.markdown("---")
    res_col, shap_col = st.columns([1, 2])

    with res_col:
        st.subheader("Result")

        # --- Verdict ---
        if prediction == 1:
            st.success("## PASS")
        else:
            st.error("## FAIL")

        c1, c2 = st.columns(2)
        c1.metric("Pass probability", f"{pass_prob}%")
        c2.metric("Fail probability", f"{fail_prob}%")
        st.caption(f"Model: **{selected_model_name}**")

        # --- Borderline warning ---
        if 45 <= pass_prob <= 55:
            st.warning(
                "⚠️ **Borderline prediction** — the model is not confident "
                f"(pass probability: {pass_prob}%). "
                "A result this close to 50% should be treated with caution. "
                "Consider reviewing the student's details or consulting "
                "additional information before making a decision."
            )

        # --- Input sanity warning ---
        if G1 == 0 and G2 == 0:
            st.warning(
                "⚠️ **Unusual input** — both G1 and G2 are 0. "
                "This is an extreme value that may produce unreliable predictions."
            )
        if absences > 50:
            st.warning(
                f"⚠️ **High absences** — {absences} absences is unusually high "
                "and may strongly skew the prediction."
            )

        # --- Best model comparison if user switched ---
        if selected_model_name != best_model_name:
            st.markdown("---")
            st.markdown("**Gradient Boosting (best model) says:**")
            best_pred  = best_model.predict(input_df)[0]
            best_proba = round(best_model.predict_proba(input_df)[0][1] * 100, 1)
            if best_pred == 1:
                st.success(f"PASS — {best_proba}% confidence")
            else:
                st.error(f"FAIL — {100 - best_proba}% confidence")

        # --- Navigation hint ---
        st.markdown("---")
        st.info(
            "Prediction saved! Go to **Explanations** to see why, "
            "or **Model Comparison** to compare all 5 models on this student."
        )

    # --- SHAP chart ---
    with shap_col:
        st.subheader("Why this prediction? (SHAP)")
        try:
            if selected_model_name in [
                "Gradient Boosting (Best)", "Random Forest", "Decision Tree"
            ]:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(input_df)
                if isinstance(shap_vals, list):
                    sv = shap_vals[1][0]
                elif np.array(shap_vals).ndim == 3:
                    sv = np.array(shap_vals)[0, :, 1]
                else:
                    sv = shap_vals[0]
            else:
                explainer = shap.Explainer(model, X_train)
                sv_obj    = explainer(input_df)
                sv        = sv_obj.values[0]
                if sv.ndim == 2:
                    sv = sv[:, 1]

            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP":    sv
            })
            shap_df = shap_df.reindex(
                shap_df["SHAP"].abs().sort_values(ascending=False).index
            ).head(10).sort_values("SHAP")

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

            # --- Plain English summary ---
            top_pos = shap_df[shap_df["SHAP"] > 0].iloc[-1]
            top_neg = shap_df[shap_df["SHAP"] < 0].iloc[0] \
                      if len(shap_df[shap_df["SHAP"] < 0]) > 0 else None

            summary = (
                f"The strongest factor pushing toward **{verdict}** is "
                f"**{top_pos['Feature']}** (SHAP: +{round(top_pos['SHAP'], 3)})"
            )
            if top_neg is not None:
                summary += (
                    f", while **{top_neg['Feature']}** works against it "
                    f"(SHAP: {round(top_neg['SHAP'], 3)})."
                )
            else:
                summary += "."

            st.caption(summary)

        except Exception as e:
            st.warning(f"SHAP explanation could not be generated: {e}")

# --------------------------------------------------
# Prediction history table
# --------------------------------------------------
if st.session_state["prediction_history"]:
    st.markdown("---")
    st.subheader("Prediction history — this session")
    st.caption("All students predicted since you opened the app.")

    history_df = pd.DataFrame(st.session_state["prediction_history"])

    # Colour the Prediction column
    def colour_verdict(val):
        if val == "PASS":
            return "background-color: #14532d; color: #86efac"
        return "background-color: #7f1d1d; color: #fca5a5"

    styled = history_df.style.applymap(colour_verdict, subset=["Prediction"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Download button
    csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download history as CSV",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

    # Clear history button
    if st.button("Clear history"):
        st.session_state["prediction_history"] = []
        st.rerun()