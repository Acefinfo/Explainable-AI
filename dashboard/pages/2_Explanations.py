import re
import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.loader import (
    load_model, load_scaler, load_train_data,
    load_test_data, get_feature_names, MODEL_OPTIONS
)

st.set_page_config(page_title="Explanations", layout="wide")
st.title("SHAP & LIME Explanations")
st.caption("Explore how the models explain their predictions — globally and for individual students.")

# --------------------------------------------------
# Load resources
# --------------------------------------------------
scaler           = load_scaler()
X_train, y_train = load_train_data()
X_test,  y_test  = load_test_data()
feature_names    = get_feature_names()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.markdown("### Controls")
selected_model_name = st.sidebar.selectbox(
    "Select model", list(MODEL_OPTIONS.keys()), index=0
)
model = load_model(MODEL_OPTIONS[selected_model_name])

# --------------------------------------------------
# Data source
# --------------------------------------------------
has_user_input = "input_df" in st.session_state

if has_user_input:
    st.info(
        "Explaining the student you entered on the Predict page. "
        "Use the option below to switch to test data instead."
    )
    use_user_input = st.radio(
        "Data to explain:",
        ["Student from Predict page", "Pick from test dataset"],
        index=0, horizontal=True
    )
else:
    st.warning(
        "No prediction made yet. Showing test dataset samples. "
        "Go to the Predict page, enter student details and click "
        "Run Prediction — then come back here to explain that student."
    )
    use_user_input = "Pick from test dataset"

if use_user_input == "Pick from test dataset" or not has_user_input:
    sample_idx   = st.sidebar.slider(
        "Test sample to explain", 0, len(X_test) - 1, 0
    )
    input_df     = X_test.iloc[[sample_idx]]
    actual_label = "PASS" if int(y_test.iloc[sample_idx]) == 1 else "FAIL"
    source_label = f"Test student #{sample_idx}"
else:
    input_df     = st.session_state["input_df"]
    actual_label = "Unknown (user input)"
    source_label = "Student from Predict page"

pred_raw   = model.predict(input_df)[0]
pred_label = "PASS" if pred_raw == 1 else "FAIL"
pred_proba = model.predict_proba(input_df)[0]
pass_pct   = round(pred_proba[1] * 100, 1)
fail_pct   = round(pred_proba[0] * 100, 1)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Source:** {source_label}")
st.sidebar.markdown(f"**Actual outcome:** {actual_label}")
st.sidebar.markdown(f"**Model predicted:** {pred_label}")
st.sidebar.metric("Pass confidence", f"{pass_pct}%")

# --------------------------------------------------
# SHAP computation
# --------------------------------------------------
def compute_shap(model, model_name, input_df, X_test, X_train, feature_names):
    try:
        if model_name in ["Gradient Boosting (Best)", "Random Forest", "Decision Tree"]:
            explainer  = shap.TreeExplainer(model)
            raw_all    = explainer.shap_values(X_test)
            raw_single = explainer.shap_values(input_df)
            base_val   = explainer.expected_value

            if isinstance(raw_all, list):
                sv_all    = raw_all[1]
                sv_single = raw_single[1][0]
                base_val  = base_val[1] if hasattr(base_val, '__len__') else base_val
            elif np.array(raw_all).ndim == 3:
                sv_all    = np.array(raw_all)[:, :, 1]
                sv_single = np.array(raw_single)[0, :, 1]
                base_val  = base_val[1] if hasattr(base_val, '__len__') else base_val
            else:
                sv_all    = raw_all
                sv_single = raw_single[0]
        else:
            explainer  = shap.Explainer(model, X_train)
            obj_all    = explainer(X_test)
            obj_single = explainer(input_df)
            sv_all     = obj_all.values[:, :, 1] if obj_all.values.ndim == 3 \
                         else obj_all.values
            sv_single  = obj_single.values[0, :, 1] if obj_single.values.ndim == 3 \
                         else obj_single.values[0]
            base_val   = explainer.expected_value

        return sv_single, sv_all, base_val, True
    except Exception as e:
        return None, None, None, str(e)


sv_single, sv_all, base_val, shap_ok = compute_shap(
    model, selected_model_name, input_df, X_test, X_train, feature_names
)
shap_computed = shap_ok is True

# --------------------------------------------------
# Dark chart helper
# --------------------------------------------------
def dark_fig(figsize=(6, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[:].set_color("#333")
    return fig, ax

# --------------------------------------------------
# Plain English helpers
# --------------------------------------------------
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

def interpret_global_shap(mean_abs_series):
    """Generate plain English summary of global SHAP importance."""
    top3 = mean_abs_series.sort_values(ascending=False).head(3)
    names = [friendly(f) for f in top3.index]
    return (
        f"Across all students in the test set, the three features that "
        f"influence the model's decisions the most are: "
        f"**{names[0]}**, **{names[1]}**, and **{names[2]}**. "
        f"This means these factors consistently matter most when predicting "
        f"whether a student will pass or fail — regardless of the individual student."
    )

def interpret_local_shap(shap_df, pred_label, source_label):
    """Generate plain English summary of local SHAP for one student."""
    positive = shap_df[shap_df["SHAP"] > 0].sort_values("SHAP", ascending=False)
    negative = shap_df[shap_df["SHAP"] < 0].sort_values("SHAP", ascending=True)

    lines = [
        f"For **{source_label}**, the model predicted **{pred_label}**. "
        f"Here is why:"
    ]

    if not positive.empty:
        top_pos = positive.iloc[0]
        lines.append(
            f"- The biggest reason pushing toward **PASS** is their "
            f"**{friendly(top_pos['Feature'])}** "
            f"(impact score: +{round(top_pos['SHAP'], 3)}). "
            + (f"Their **{friendly(positive.iloc[1]['Feature'])}** also helped."
               if len(positive) > 1 else "")
        )

    if not negative.empty:
        top_neg = negative.iloc[0]
        lines.append(
            f"- The biggest factor working against them is their "
            f"**{friendly(top_neg['Feature'])}** "
            f"(impact score: {round(top_neg['SHAP'], 3)}). "
            + (f"Their **{friendly(negative.iloc[1]['Feature'])}** also reduced the score."
               if len(negative) > 1 else "")
        )

    if positive.empty:
        lines.append("- No features are strongly pushing toward a PASS for this student.")
    if negative.empty:
        lines.append("- No features are working against this student.")

    return "\n\n".join(lines)

def interpret_lime(lime_df, pred_label, source_label):
    """Generate plain English summary of LIME explanation."""
    positive = lime_df[lime_df["Weight"] > 0].sort_values("Weight", ascending=False)
    negative = lime_df[lime_df["Weight"] < 0].sort_values("Weight", ascending=True)

    lines = [
        f"LIME looked closely at **{source_label}** and found:"
    ]

    if not positive.empty:
        cond = positive.iloc[0]["Feature condition"]
        w    = round(positive.iloc[0]["Weight"], 3)
        lines.append(
            f"- The condition **\"{cond}\"** is the strongest reason "
            f"pushing toward **PASS** (weight: +{w})."
        )
    if not negative.empty:
        cond = negative.iloc[0]["Feature condition"]
        w    = round(negative.iloc[0]["Weight"], 3)
        lines.append(
            f"- The condition **\"{cond}\"** is the strongest factor "
            f"working against the student (weight: {w})."
        )

    lines.append(
        f"Overall LIME agrees with the model's prediction of **{pred_label}**. "
        f"Note that LIME explains one student at a time — it does not describe "
        f"general model behaviour."
    )
    return "\n\n".join(lines)

def interpret_comparison(corr, pval, shap_top, lime_top):
    """Generate plain English interpretation of SHAP vs LIME agreement."""
    shared = set(shap_top.index[:5]) & set(lime_top.index[:5])
    level  = "strongly" if corr > 0.6 else "moderately" if corr > 0.3 else "weakly"

    lines = [
        f"**What does a Spearman correlation of {round(corr, 3)} mean?**",
        f"A score of 1.0 means perfect agreement, 0 means random. "
        f"Your score of **{round(corr, 3)}** means SHAP and LIME **{level} agree** "
        f"on which features matter most for this student.",
    ]

    if shared:
        shared_friendly = [friendly(f) for f in shared]
        lines.append(
            f"Both methods agree that the following features are important: "
            f"**{', '.join(shared_friendly)}**."
        )
    else:
        lines.append(
            "The two methods do not share any features in their top 5 — "
            "this suggests LIME's local approximation diverges significantly "
            "from SHAP's calculation for this student."
        )

    if pval < 0.05:
        lines.append(
            f"The p-value of {round(pval, 4)} confirms this agreement is "
            f"statistically significant — it is unlikely to be due to chance."
        )
    else:
        lines.append(
            "However, the p-value suggests this agreement may not be "
            "statistically reliable for this particular student."
        )

    lines.append(
        "**Key difference:** SHAP always gives the same answer for the same input. "
        "LIME uses random sampling, so its answer can vary slightly each time — "
        "this is why SHAP is considered more consistent for sensitive decisions."
    )

    return "\n\n".join(lines)

def interpretation_box(text):
    """Render a styled plain English interpretation box."""
    def markdown_to_html_bold(value):
        return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)

    safe_text = markdown_to_html_bold(text).replace(chr(10), "<br>")
    st.markdown(
        f"""<div style="background:#0f2027;border-left:3px solid #3b82f6;
                        border-radius:0 8px 8px 0;padding:14px 16px;
                        margin-top:10px;">
                <div style="font-size:11px;font-weight:600;color:#93c5fd;
                            margin-bottom:6px;text-transform:uppercase;
                            letter-spacing:0.05em;">
                    What does this mean?
                </div>
                <div style="font-size:13px;color:#cbd5e1;line-height:1.7;">
                    {safe_text}
                </div>
            </div>""",
        unsafe_allow_html=True
    )


# ==================== TABS ====================
tab1, tab2, tab3 = st.tabs(["SHAP", "LIME", "SHAP vs LIME"])


# ── TAB 1: SHAP ──
with tab1:
    st.subheader("SHAP Explanations")
    st.caption(
        "SHAP tells us how much each feature pushed the prediction toward "
        "Pass (blue) or Fail (red), and by how much."
    )

    if not shap_computed:
        st.error(f"Could not compute SHAP values: {shap_ok}")
    else:
        col_g, col_l = st.columns(2)

        with col_g:
            st.markdown("**Global — which features matter most overall?**")
            mean_abs = pd.Series(
                np.abs(sv_all).mean(axis=0), index=feature_names
            ).sort_values(ascending=True).tail(12)

            fig, ax = dark_fig()
            ax.barh(mean_abs.index, mean_abs.values, color="#3b82f6")
            ax.set_xlabel("Mean |SHAP value|", color="white")
            ax.set_title("Top 12 most important features", color="white", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            interpretation_box(interpret_global_shap(mean_abs))

        with col_l:
            st.markdown(f"**Local — why did the model decide this for {source_label}?**")
            shap_df = pd.DataFrame({
                "Feature": feature_names, "SHAP": sv_single
            }).reindex(
                pd.Series(sv_single).abs().sort_values(ascending=False).index
            ).head(10).sort_values("SHAP")

            colors = ["#ef4444" if v < 0 else "#3b82f6" for v in shap_df["SHAP"]]
            fig, ax = dark_fig()
            ax.barh(shap_df["Feature"], shap_df["SHAP"], color=colors)
            ax.axvline(0, color="white", linewidth=0.8)
            ax.set_xlabel("SHAP value", color="white")
            ax.set_title(
                f"{source_label} — Predicted: {pred_label} | Actual: {actual_label}",
                color="white", fontsize=9
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            interpretation_box(interpret_local_shap(shap_df, pred_label, source_label))

        st.markdown("---")
        st.markdown("**Feature contributions table**")
        top_feats  = shap_df["Feature"].tolist()
        raw_vals   = input_df.iloc[0]
        display_df = pd.DataFrame({
            "Feature":         top_feats,
            "Scaled value":    [round(float(raw_vals[f]), 3) for f in top_feats],
            "SHAP value":      [round(sv_single[feature_names.index(f)], 4)
                                for f in top_feats],
            "Direction":       ["→ Pass" if sv_single[feature_names.index(f)] > 0
                                else "→ Fail" for f in top_feats],
            "Plain English":   [friendly(f) for f in top_feats],
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ── TAB 2: LIME ──
with tab2:
    st.subheader("LIME Explanations")
    st.caption(
        "LIME zooms in on one student at a time and asks: "
        "which conditions most influenced this specific prediction?"
    )

    num_features = st.slider("Number of features to show", 5, 15, 10)

    try:
        lime_explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=feature_names,
            class_names=["Fail", "Pass"],
            mode="classification",
            random_state=42
        )
        exp = lime_explainer.explain_instance(
            input_df.iloc[0].values,
            model.predict_proba,
            num_features=num_features
        )

        lime_list = exp.as_list()
        lime_df   = pd.DataFrame({
            "Feature condition": [i[0] for i in lime_list],
            "Weight":            [i[1] for i in lime_list]
        }).sort_values("Weight")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(f"**LIME explanation — {source_label}**")
            colors = ["#ef4444" if v < 0 else "#3b82f6" for v in lime_df["Weight"]]
            fig, ax = dark_fig()
            ax.barh(lime_df["Feature condition"], lime_df["Weight"], color=colors)
            ax.axvline(0, color="white", linewidth=0.8)
            ax.set_xlabel("LIME weight", color="white")
            ax.set_title(
                f"{source_label} — Predicted: {pred_label} | Actual: {actual_label}",
                color="white", fontsize=9
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            interpretation_box(interpret_lime(lime_df, pred_label, source_label))

        with col_b:
            st.markdown("**LIME weights table**")
            st.caption(
                "Each row is a condition about this student. "
                "Positive weight → pushes toward Pass. "
                "Negative weight → pushes toward Fail."
            )
            st.dataframe(
                lime_df.assign(Weight=lime_df["Weight"].round(4)),
                use_container_width=True, hide_index=True
            )
            st.markdown("---")
            st.markdown("**Prediction probabilities**")
            st.dataframe(pd.DataFrame({
                "Outcome":     ["Fail", "Pass"],
                "Probability": [f"{fail_pct}%", f"{pass_pct}%"]
            }), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"LIME explanation failed: {e}")


# ── TAB 3: SHAP vs LIME ──
with tab3:
    st.subheader("SHAP vs LIME — Side-by-side Comparison")
    st.caption(
        "Do both methods agree on what matters? "
        "Spearman correlation measures how similar their feature rankings are."
    )

    if not shap_computed:
        st.error("SHAP values could not be computed — comparison unavailable.")
    else:
        try:
            lime_exp2 = LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=feature_names,
                class_names=["Fail", "Pass"],
                mode="classification",
                random_state=42
            ).explain_instance(
                input_df.iloc[0].values,
                model.predict_proba,
                num_features=len(feature_names)
            )

            def map_lime(exp, feature_names):
                weights = {}
                for cond, w in exp.as_list():
                    for f in feature_names:
                        if f in cond:
                            weights[f] = abs(w)
                            break
                return pd.Series(weights).reindex(feature_names, fill_value=0)

            lime_series = map_lime(lime_exp2, feature_names)
            shap_series = pd.Series(np.abs(sv_single), index=feature_names)
            corr, pval  = spearmanr(
                shap_series.rank(ascending=False),
                lime_series.rank(ascending=False)
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**SHAP — top 10 features**")
                sp = shap_series.sort_values(ascending=False).head(10).sort_values()
                fig, ax = dark_fig()
                ax.barh(sp.index, sp.values, color="#3b82f6")
                ax.set_xlabel("|SHAP value|", color="white")
                ax.set_title("SHAP", color="white")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                st.markdown("**LIME — top 10 features**")
                lp = lime_series.sort_values(ascending=False).head(10).sort_values()
                fig, ax = dark_fig()
                ax.barh(lp.index, lp.values, color="#f59e0b")
                ax.set_xlabel("|LIME weight|", color="white")
                ax.set_title("LIME", color="white")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            # Agreement metrics
            st.markdown("---")
            st.markdown("**Agreement between SHAP and LIME**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Spearman correlation", f"{round(corr, 3)}")
            m2.metric("P-value", f"{round(pval, 4)}")
            m3.metric(
                "Agreement level",
                "Strong" if corr > 0.6 else "Moderate" if corr > 0.3 else "Weak"
            )

            # Plain English interpretation
            shap_top10 = shap_series.sort_values(ascending=False).head(10)
            lime_top10 = lime_series.sort_values(ascending=False).head(10)
            interpretation_box(
                interpret_comparison(corr, pval, shap_top10, lime_top10)
            )

            # Rank comparison table
            st.markdown("---")
            st.markdown("**Feature rank comparison table**")
            st.caption(
                "Rank 1 = most important. "
                "Compare the SHAP and LIME columns to see where they agree and disagree."
            )
            st.dataframe(pd.DataFrame({
                "SHAP rank":     range(1, 11),
                "SHAP feature":  shap_top10.index.tolist(),
                "SHAP |value|":  shap_top10.values.round(4),
                "LIME rank":     range(1, 11),
                "LIME feature":  lime_top10.index.tolist(),
                "LIME |weight|": lime_top10.values.round(4),
            }), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Comparison failed: {e}")