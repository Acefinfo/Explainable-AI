import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.loader import (
    load_model, load_cv_results, load_spearman_results,
    get_feature_names, MODEL_OPTIONS
)

st.set_page_config(page_title="Model Comparison", page_icon="", layout="wide")
st.title("Model Comparison")
st.caption("Compare how all 5 models respond to the same student input.")

# --------------------------------------------------
# Check session state
# --------------------------------------------------
if "input_df" not in st.session_state:
    st.warning(
        "No student data found. Please go to the **Predict** page, "
        "fill in the student details and click **Run Prediction** first — "
        "then come back here to compare all models on that student."
    )
    st.stop()

input_df     = st.session_state["input_df"]
active_model = st.session_state.get("model_name", "Gradient Boosting (Best)")

st.info(
    f"Comparing all 5 models on the student you entered on the Predict page. "
    f"Your selected model was: **{active_model}**"
)

feature_names = get_feature_names()

# --------------------------------------------------
# Run all 5 models on the user's input
# --------------------------------------------------
st.markdown("---")
st.subheader("All models predict on your student")

model_results = []
trained_models = {}

for model_name, filename in MODEL_OPTIONS.items():
    mdl        = load_model(filename)
    pred       = mdl.predict(input_df)[0]
    proba      = mdl.predict_proba(input_df)[0]
    pass_prob  = round(proba[1] * 100, 1)
    fail_prob  = round(proba[0] * 100, 1)
    verdict    = "PASS" if pred == 1 else "FAIL"
    trained_models[model_name] = mdl
    model_results.append({
        "Model":            model_name,
        "Prediction":       verdict,
        "Pass probability": pass_prob,
        "Fail probability": fail_prob,
        "Confidence":       pass_prob if pred == 1 else fail_prob,
        "Your selection":   "✓" if model_name == active_model else ""
    })

results_df = pd.DataFrame(model_results)

# --------------------------------------------------
# Verdict cards — one per model
# --------------------------------------------------
cols = st.columns(len(MODEL_OPTIONS))
short_names = {
    "Gradient Boosting (Best)": "Gradient\nBoosting",
    "Random Forest":            "Random\nForest",
    "Logistic Regression":      "Logistic\nRegression",
    "Decision Tree":            "Decision\nTree",
    "Support Vector Machine":   "SVM",
}

cols = st.columns(len(MODEL_OPTIONS))

for col, row in zip(cols, model_results):
    with col:
        is_active  = row["Model"] == active_model
        border     = "2px solid #3b82f6" if is_active else "1px solid #333"
        bg         = "#1a2a3a" if is_active else "#1a1a1a"
        verdict    = row["Prediction"]
        colour     = "#22c55e" if verdict == "PASS" else "#ef4444"
        star       = "⭐" if is_active else ""
        name_clean = short_names[row["Model"]].replace("\n", "<br>")
        pass_p     = row["Pass probability"]

        col.markdown(
            f"""<div style="border:{border};background:{bg};border-radius:10px;
                            padding:14px;text-align:center;min-height:120px;">
                    <div style="font-size:11px;color:#94a3b8;margin-bottom:6px;">
                        {name_clean} {star}
                    </div>
                    <div style="font-size:26px;font-weight:700;color:{colour};
                                margin-bottom:4px;">
                        {verdict}
                    </div>
                    <div style="font-size:13px;color:#94a3b8;">
                        {pass_p}% pass confidence
                    </div>
                </div>""",
            unsafe_allow_html=True
)
# --------------------------------------------------
# Confidence bar chart — all models
# --------------------------------------------------
st.markdown("---")
st.subheader("Pass probability comparison")

fig, ax = plt.subplots(figsize=(9, 3.5))
fig.patch.set_facecolor("#0e1117")
ax.set_facecolor("#0e1117")

model_labels = [short_names[r["Model"]].replace("\n", " ") for r in model_results]
pass_probs   = [r["Pass probability"] for r in model_results]
bar_colors   = [
    "#3b82f6" if r["Model"] == active_model else
    "#22c55e" if r["Pass probability"] >= 50 else "#ef4444"
    for r in model_results
]

bars = ax.bar(model_labels, pass_probs, color=bar_colors,
              edgecolor="#555", linewidth=0.5, width=0.5)
ax.axhline(50, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_ylabel("Pass probability (%)", color="white")
ax.set_ylim(0, 110)
ax.tick_params(colors="white", labelsize=9)
ax.spines[:].set_color("#333")
ax.text(len(model_labels) - 0.5, 52, "50% threshold",
        color="white", fontsize=8, alpha=0.6)

for bar, val in zip(bars, pass_probs):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val}%", ha="center", va="bottom",
            color="white", fontsize=9, fontweight="bold")

ax.legend(
    handles=[
        plt.Rectangle((0,0),1,1, color="#3b82f6", label="Your selected model"),
        plt.Rectangle((0,0),1,1, color="#22c55e", label="Predicts PASS"),
        plt.Rectangle((0,0),1,1, color="#ef4444", label="Predicts FAIL"),
    ],
    facecolor="#1a1a1a", labelcolor="white", fontsize=8,
    loc="upper right"
)

plt.tight_layout()
st.pyplot(fig)
plt.close()

# --------------------------------------------------
# Detailed results table
# --------------------------------------------------
st.markdown("---")
st.subheader("Detailed results table")

display_df = results_df[[
    "Model", "Prediction", "Pass probability",
    "Fail probability", "Your selection"
]].copy()

st.dataframe(display_df, use_container_width=True, hide_index=True)

# --------------------------------------------------
# Agreement analysis
# --------------------------------------------------
st.markdown("---")
st.subheader("Model agreement")

verdicts    = [r["Prediction"] for r in model_results]
pass_count  = verdicts.count("PASS")
fail_count  = verdicts.count("FAIL")
majority    = "PASS" if pass_count >= fail_count else "FAIL"
all_agree   = pass_count == 5 or fail_count == 5

a1, a2, a3 = st.columns(3)
a1.metric("Models predicting PASS", pass_count)
a2.metric("Models predicting FAIL", fail_count)
a3.metric("Majority verdict", majority)

if all_agree:
    st.success(
        f"All 5 models agree — this student is predicted to **{majority}**. "
        f"This is a high-confidence result."
    )
elif pass_count >= 4 or fail_count >= 4:
    minority = [r["Model"] for r in model_results if r["Prediction"] != majority]
    st.info(
        f"Strong agreement — 4 out of 5 models predict **{majority}**. "
        f"Only **{minority[0]}** disagrees."
    )
else:
    disagreeing = [r["Model"] for r in model_results if r["Prediction"] != majority]
    st.warning(
        f"Mixed predictions — majority ({pass_count if majority=='PASS' else fail_count}/5) "
        f"predict **{majority}**, but {', '.join(disagreeing)} disagree. "
        f"Treat this prediction with caution."
    )

# --------------------------------------------------
# SHAP comparison — all models on same input
# --------------------------------------------------
st.markdown("---")
st.subheader("SHAP feature importance per model")
st.caption("Which features drive each model's decision for this specific student?")

tree_models = {
    k: v for k, v in trained_models.items()
    if k in ["Gradient Boosting (Best)", "Random Forest", "Decision Tree"]
}

shap_cols = st.columns(len(tree_models))

for col, (mname, mdl) in zip(shap_cols, tree_models.items()):
    with col:
        try:
            explainer  = shap.TreeExplainer(mdl)
            raw        = explainer.shap_values(input_df)

            if isinstance(raw, list):
                sv = raw[1][0]
            elif np.array(raw).ndim == 3:
                sv = np.array(raw)[0, :, 1]
            else:
                sv = raw[0]

            shap_df = pd.DataFrame({
                "Feature": feature_names, "SHAP": sv
            }).reindex(
                pd.Series(sv).abs().sort_values(ascending=False).index
            ).head(8).sort_values("SHAP")

            colors = ["#ef4444" if v < 0 else "#3b82f6" for v in shap_df["SHAP"]]

            fig, ax = plt.subplots(figsize=(4, 3.5))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")
            ax.barh(shap_df["Feature"], shap_df["SHAP"], color=colors)
            ax.axvline(0, color="white", linewidth=0.6)
            ax.set_xlabel("SHAP", color="white", fontsize=8)
            ax.set_title(
                short_names[mname].replace("\n", " "),
                color="white", fontsize=9
            )
            ax.tick_params(colors="white", labelsize=7)
            ax.spines[:].set_color("#333")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.warning(f"{mname}: {e}")

# --------------------------------------------------
# CV performance context
# --------------------------------------------------
st.markdown("---")
st.subheader("Overall model performance (5-fold CV)")
st.caption("For context — how each model performs on the full dataset, not just this student.")

try:
    cv_df = load_cv_results()
    st.dataframe(cv_df, use_container_width=True, hide_index=True)
except Exception:
    st.info("CV results file not found. Run the cross-validation notebook first.")

# --------------------------------------------------
# Spearman context
# --------------------------------------------------
st.markdown("---")
st.subheader("SHAP vs LIME agreement (from notebook analysis)")

try:
    sp_df = load_spearman_results()
    m1, m2 = st.columns(2)
    m1.metric(
        "Mean Spearman correlation",
        f"{round(sp_df['Spearman Correlation'].mean(), 3)}"
    )
    m2.metric(
        "Overall agreement level",
        "Moderate" if sp_df['Spearman Correlation'].mean() > 0.3 else "Weak"
    )
    st.dataframe(sp_df, use_container_width=True, hide_index=True)
except Exception:
    st.info("Spearman results file not found. Run notebook 07 first.")