import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from scipy.stats import entropy

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.loader import (
    load_model, load_cv_results, load_spearman_results,
    get_feature_names, MODEL_OPTIONS
)

st.set_page_config(page_title="Model Comparison", page_icon="", layout="wide")
st.title("Model Comparison & Explanations")
st.caption("Compare all 5 models AND see WHY they make different decisions for your student.")

# ──────────────────────────────────────────────────────────────────────────────
# CHECK SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
if "input_df" not in st.session_state:
    st.warning(
        "No student data found. Please go to the **Predict** page, "
        "fill in the student details and click **Run Prediction** first — "
        "then come back here to compare all models on that student."
    )
    st.stop()

input_df     = st.session_state["input_df"]
active_model = st.session_state.get("model_name", "Gradient Boosting (Best)")
feature_names = get_feature_names()

st.info(
    f"Comparing all 5 models on the student you entered. "
    f"Your selected model: **{active_model}**"
)

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE LABEL HELPER (same as in Explanations page)
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_LABELS = {
    "G1":             "first period grade (G1)",
    "G2":             "second period grade (G2)",
    "failures":       "number of past failures",
    "absences":       "number of absences",
    "studytime":      "weekly study time",
    "goout":          "how often they go out",
    "health":         "health status",
    "age":            "age",
    "Medu":           "mother's education level",
    "Fedu":           "father's education level",
    "famrel":         "family relationship quality",
    "higher_yes":     "wanting higher education",
    "internet_yes":   "having internet access",
    "romantic_yes":   "being in a romantic relationship",
    "schoolsup_yes":  "receiving extra school support",
}

def friendly(feature):
    """Return a plain English label for a feature name."""
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))

def interpretation_box(text):
    """Display interpretation in a nice box."""
    st.markdown(
        f"""
        <div style="background-color:#1e3a5f;border-left:4px solid #3b82f6;padding:12px;border-radius:4px;margin:12px 0;">
            <p style="color:#e0e7ff;margin:0;font-size:14px;">{text}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────────────────────────────────────────
# RUN ALL 5 MODELS
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# VERDICT CARDS — ONE PER MODEL
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# CONFIDENCE BAR CHART
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# AGREEMENT ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Model Agreement Analysis")

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
    interpretation_box(
        f"✅ **Full Consensus:** All 5 models agree — this student is predicted to **{majority}**. "
        f"This is a very high-confidence result with very low uncertainty."
    )
elif pass_count >= 4 or fail_count >= 4:
    minority = [r["Model"] for r in model_results if r["Prediction"] != majority]
    interpretation_box(
        f"✅ **Strong Agreement:** 4 out of 5 models predict **{majority}**. "
        f"Only **{minority[0]}** disagrees. This is a high-confidence prediction."
    )
else:
    disagreeing = [r["Model"] for r in model_results if r["Prediction"] != majority]
    interpretation_box(
        f"⚠️ **Mixed Predictions:** Majority ({pass_count if majority=='PASS' else fail_count}/5) "
        f"predict **{majority}**, but {', '.join(disagreeing)} disagree. "
        f"This student is on the decision boundary. Treat this prediction with caution and consider additional factors."
    )

# ──────────────────────────────────────────────────────────────────────────────
# PER-MODEL EXPLANATIONS WITH SHAP
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Why Does Each Model Make Its Decision?")
st.caption("Hover over sections to see which features influenced each model's decision.")

# Create tabs for each model
model_tabs = st.tabs([short_names[r["Model"]].replace("\n", " ") for r in model_results])

for tab, row, (model_name, mdl) in zip(model_tabs, model_results, trained_models.items()):
    with tab:
        pred_verdict = row["Prediction"]
        pass_prob = row["Pass probability"]
        confidence = row["Confidence"]
        
        # Prediction summary
        verdict_color = "🟢 PASS" if pred_verdict == "PASS" else "🔴 FAIL"
        st.markdown(f"### {verdict_color} ({confidence}% confidence)")
        
        # Try to compute SHAP values
        try:
            if model_name in ["Gradient Boosting (Best)", "Random Forest", "Decision Tree"]:
                explainer = shap.TreeExplainer(mdl)
                raw_shap = explainer.shap_values(input_df)
                base_val = explainer.expected_value
                
                # Handle different return formats
                if isinstance(raw_shap, list):
                    sv = raw_shap[1][0]
                    bv = base_val[1] if hasattr(base_val, '__len__') else base_val
                elif np.array(raw_shap).ndim == 3:
                    sv = np.array(raw_shap)[0, :, 1]
                    bv = base_val[1] if hasattr(base_val, '__len__') else base_val
                else:
                    sv = raw_shap[0]
                    bv = base_val
            else:
                # For non-tree models, use kernel SHAP
                explainer = shap.Explainer(mdl, input_df)
                obj = explainer(input_df)
                sv = obj.values[0, :, 1] if obj.values.ndim == 3 else obj.values[0]
                bv = explainer.expected_value
            
            # Create SHAP dataframe for this model
            shap_df = pd.DataFrame({
                "Feature": feature_names, 
                "SHAP": sv
            }).reindex(
                pd.Series(sv).abs().sort_values(ascending=False).index
            ).head(10).sort_values("SHAP")
            
            # Visualize top features
            col_chart, col_table = st.columns([1.2, 1])
            
            with col_chart:
                st.markdown("**Feature contributions (top 10)**")
                colors = ["#ef4444" if v < 0 else "#3b82f6" for v in shap_df["SHAP"]]
                
                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_facecolor("#0e1117")
                ax.set_facecolor("#0e1117")
                ax.barh(shap_df["Feature"], shap_df["SHAP"], color=colors)
                ax.axvline(0, color="white", linewidth=0.8)
                ax.set_xlabel("SHAP value", color="white", fontsize=9)
                ax.set_title(f"{short_names[model_name]}", color="white", fontsize=10)
                ax.tick_params(colors="white", labelsize=8)
                ax.spines[:].set_color("#333")
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with col_table:
                st.markdown("**Feature impact details**")
                display_df = pd.DataFrame({
                    "Feature": shap_df["Feature"].apply(friendly),
                    "Impact": shap_df["SHAP"].apply(lambda x: f"{x:+.4f}"),
                    "Direction": ["→ Pass" if sv > 0 else "→ Fail" 
                                 for sv in shap_df["SHAP"]],
                    "Value": [round(float(input_df.iloc[0][f]), 2) 
                             for f in shap_df["Feature"]]
                })
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Plain English interpretation
            st.markdown("---")
            st.markdown("**Plain English explanation**")
            
            positive = shap_df[shap_df["SHAP"] > 0].sort_values("SHAP", ascending=False)
            negative = shap_df[shap_df["SHAP"] < 0].sort_values("SHAP", ascending=True)
            
            explanation_text = f"This model predicts **{pred_verdict}** ({confidence}% confidence). "
            
            if not positive.empty:
                top_pos = positive.iloc[0]
                explanation_text += (
                    f"The strongest factor pushing toward **PASS** is their "
                    f"**{friendly(top_pos['Feature'])}** (impact: +{round(top_pos['SHAP'], 3)}). "
                )
                if len(positive) > 1:
                    explanation_text += (
                        f"Their **{friendly(positive.iloc[1]['Feature'])}** also helps. "
                    )
            
            if not negative.empty:
                top_neg = negative.iloc[0]
                explanation_text += (
                    f"The biggest concern is their **{friendly(top_neg['Feature'])}** "
                    f"(impact: {round(top_neg['SHAP'], 3)}). "
                )
                if len(negative) > 1:
                    explanation_text += (
                        f"Their **{friendly(negative.iloc[1]['Feature'])}** also reduces the pass score."
                    )
            
            interpretation_box(explanation_text)
            
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {str(e)}")
            st.info(f"This model type may not be fully compatible with SHAP. "
                   f"Showing prediction only: **{pred_verdict}** ({confidence}% confidence)")

# ──────────────────────────────────────────────────────────────────────────────
# WHY DO MODELS DISAGREE?
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Why Do Models Sometimes Disagree?")

if all_agree:
    st.success("All models agree! No disagreement to analyze.")
else:
    explanation = """
    Different models make different decisions for the same student because they **learn different patterns**:
    
    1. **Different Algorithms:**
       - **Tree-based models** (Random Forest, Decision Tree, Gradient Boosting) make predictions by splitting data on features
       - **Logistic Regression** uses a linear boundary
       - **SVM** uses a non-linear boundary with different kernel
    
    2. **Different Decision Boundaries:**
       - Each model learns where the PASS/FAIL boundary is differently
       - A student right on the boundary may fall on different sides for different models
    
    3. **Feature Sensitivity:**
       - Some models weight features heavily; others don't
       - Tree models can interact features; linear models cannot
    
    4. **Training Variance:**
       - Even with the same data, different algorithms converge to different solutions
       - This is normal and expected!
    
    **What should you do?**
    - When models disagree, it's a **yellow flag** — investigate further
    - Consider the student's overall profile, not just one model
    - Look at the features that models agree on (most important indicators)
    """
    
    st.markdown(explanation)

# ──────────────────────────────────────────────────────────────────────────────
# MODEL CONSENSUS ON TOP FEATURES
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("What Features Matter Most ACROSS All Models?")
st.caption("Even when models disagree on predictions, they often agree on which features matter.")

# Compute feature importance agreement across models
feature_importance_agreement = {}

for model_name, mdl in trained_models.items():
    try:
        if model_name in ["Gradient Boosting (Best)", "Random Forest", "Decision Tree"]:
            explainer = shap.TreeExplainer(mdl)
            raw_shap = explainer.shap_values(input_df)
            
            if isinstance(raw_shap, list):
                sv = np.abs(raw_shap[1][0])
            elif np.array(raw_shap).ndim == 3:
                sv = np.abs(np.array(raw_shap)[0, :, 1])
            else:
                sv = np.abs(raw_shap[0])
        else:
            explainer = shap.Explainer(mdl, input_df)
            obj = explainer(input_df)
            sv = np.abs(obj.values[0, :, 1] if obj.values.ndim == 3 else obj.values[0])
        
        feature_importance_agreement[model_name] = sv
    except:
        pass

if feature_importance_agreement:
    # Create consensus ranking
    avg_importance = np.zeros(len(feature_names))
    for sv in feature_importance_agreement.values():
        avg_importance += sv
    avg_importance /= len(feature_importance_agreement)
    
    consensus_df = pd.DataFrame({
        "Feature": feature_names,
        "Average Importance": avg_importance,
        "Rank": pd.Series(avg_importance).rank(ascending=False, method='dense').values.astype(int)
    }).sort_values("Average Importance", ascending=False).head(5)
    
    consensus_df["Feature Name"] = consensus_df["Feature"].apply(friendly)
    
    st.markdown("**Top 5 features that models agree on:**")
    
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    
    consensus_sort = consensus_df.sort_values("Average Importance")
    ax.barh(consensus_sort["Feature Name"], consensus_sort["Average Importance"], 
            color="#10b981", alpha=0.8)
    ax.set_xlabel("Average SHAP |value| across all models", color="white", fontsize=9)
    ax.set_title("Model Consensus: Most Important Features", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[:].set_color("#333")
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Show as table
    st.dataframe(
        consensus_df[["Rank", "Feature Name", "Average Importance"]].rename(
            columns={"Feature Name": "Feature", "Average Importance": "Consensus Score"}
        ),
        use_container_width=True, hide_index=True
    )

# ──────────────────────────────────────────────────────────────────────────────
# DETAILED RESULTS TABLE
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Detailed Results Table")

display_df = results_df[[
    "Model", "Prediction", "Pass probability",
    "Fail probability", "Your selection"
]].copy()

st.dataframe(display_df, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
# CV PERFORMANCE CONTEXT
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Overall Model Performance (5-fold CV)")
st.caption("For context — how each model performs on the full dataset, not just this student.")

try:
    cv_df = load_cv_results()
    st.dataframe(cv_df, use_container_width=True, hide_index=True)
except Exception:
    st.info("CV results file not found. Run the cross-validation notebook first.")

# ──────────────────────────────────────────────────────────────────────────────
# EXPLANATION METHOD COMPARISON
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("SHAP vs LIME Agreement")
st.caption("How well do different explanation methods agree on this student?")

try:
    sp_df = load_spearman_results()
    m1, m2 = st.columns(2)
    m1.metric(
        "Mean Spearman correlation",
        f"{round(sp_df['Spearman Correlation'].mean(), 3)}"
    )
    m2.metric(
        "Overall agreement level",
        "Strong" if sp_df['Spearman Correlation'].mean() > 0.6 
        else "Moderate" if sp_df['Spearman Correlation'].mean() > 0.3 
        else "Weak"
    )
    st.dataframe(sp_df, use_container_width=True, hide_index=True)
    
    # Interpretation
    mean_corr = sp_df['Spearman Correlation'].mean()
    if mean_corr > 0.6:
        agreement_text = (
            "SHAP and LIME **strongly agree** on which features matter. "
            "This is excellent — both methods reach consistent conclusions."
        )
    elif mean_corr > 0.3:
        agreement_text = (
            "SHAP and LIME **moderately agree**. They identify similar important features "
            "but may rank them differently. Use both methods to get a complete picture."
        )
    else:
        agreement_text = (
            "SHAP and LIME **weakly agree**. They identify different important features. "
            "This suggests explanation results are sensitive to the method used. "
            "Look at both explanations carefully."
        )
    
    interpretation_box(agreement_text)
    
except Exception:
    st.info("Spearman results file not found. Run notebook 07 first.")

# ──────────────────────────────────────────────────────────────────────────────
# FINAL RECOMMENDATIONS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Recommendations Based on Model Comparison")

verdicts = [r["Prediction"] for r in model_results]
pass_count_rec = verdicts.count("PASS")
fail_count_rec = verdicts.count("FAIL")
majority_rec = "PASS" if pass_count_rec >= fail_count_rec else "FAIL"

if all_agree:
    rec_text = (
        f"**Strong Signal:** All models agree on **{majority_rec}**. "
        f"You can be confident in this prediction. The key factors are consistent across all models."
    )
elif pass_count_rec >= 4 or fail_count_rec >= 4:
    rec_text = (
        f"**Follow the Majority:** 4 out of 5 models predict **{majority_rec}**. "
        f"This is a reliable prediction, though one model disagrees."
    )
else:
    rec_text = (
        f"**Borderline Case:** Models split on this prediction. "
        f"Review the student's profile on factors all models agree matter (see 'Model Consensus' section). "
        f"This student is at the boundary and could go either way."
    )

interpretation_box(rec_text)

st.caption(
    "**Pro tip:** Use the Explanations page to see detailed SHAP and LIME explanations "
    "for even deeper insight into why predictions are made."
)