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

st.set_page_config(page_title="Explanations",  layout="wide")
st.title("SHAP & LIME Explanations")
st.caption("Explore how the models explain their predictions — globally and for individual students.")

# --------------------------------------------------
# Load resources(Data and models)
# --------------------------------------------------
scaler           = load_scaler()
X_train, y_train = load_train_data()
X_test,  y_test  = load_test_data()
feature_names    = get_feature_names()

# --------------------------------------------------
# Sidebar — model selector
# --------------------------------------------------
st.sidebar.markdown("### Controls")

selected_model_name = st.sidebar.selectbox(
    "Select model",
    list(MODEL_OPTIONS.keys()),
    index=0
)
model = load_model(MODEL_OPTIONS[selected_model_name])

# --------------------------------------------------
# Decide data source — session state or test set
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
        index=0,
        horizontal=True
    )
else:
    st.warning(
        "No prediction made yet. Showing test dataset samples. "
        "Go to the Predict page, enter student details and click "
        "Run Prediction — then come back here to explain that student."
    )
    use_user_input = "Pick from test dataset"

# Output for the explanation when the user had no input and is trying to explore the models understanding.
if use_user_input == "Pick from test dataset" or not has_user_input:
    sample_idx = st.sidebar.slider(
        "Test sample to explain",
        min_value=0, max_value=len(X_test) - 1, value=0
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

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Source:** {source_label}")
st.sidebar.markdown(f"**Actual outcome:** {actual_label}")
st.sidebar.markdown(f"**Model predicted:** {pred_label}")
st.sidebar.metric("Pass confidence", f"{round(pred_proba[1]*100, 1)}%")

# --------------------------------------------------
# Compute SHAP values — fixed array handling
# --------------------------------------------------
def compute_shap(model, model_name, input_df, X_test, X_train, feature_names):
    """
    Compute SHAP values for both global (dataset-level) and local (single instance)
    explanations, with support for multiple model types and SHAP output formats.

    This function handles:
    - Tree-based models using TreeExplainer (optimized and faster)
    - Non-tree models using the general SHAP Explainer
    - Different SHAP output formats across versions (list, 2D, 3D arrays)
    - Binary classification by extracting SHAP values for the positive class (index 1)

    Parameters:
        model : object
            Trained machine learning model used for prediction.

        model_name : str
            Name of the selected model. Used to determine whether to apply
            TreeExplainer (for tree-based models) or the general Explainer.

        input_df : pandas.DataFrame
            Single input sample (1 row) for which local SHAP explanation is computed.

        X_test : pandas.DataFrame
            Test dataset used to compute global SHAP values.

        X_train : pandas.DataFrame
            Training dataset required for initializing the general SHAP Explainer
            (used for non-tree models).

        feature_names : list
            List of feature names (not directly used in computation but kept for consistency).

    Returns:
        sv_single : numpy.ndarray or None
            SHAP values for a single input sample (1D array of feature contributions).
            Returns None if computation fails.

        sv_all : numpy.ndarray or None
            SHAP values for the entire test dataset (2D array: samples x features).
            Returns None if computation fails.

        base_val : float or None
            Expected value (baseline prediction) from the SHAP explainer.
            Represents the average model output before feature contributions.

        status : bool or str
            - True → SHAP computation successful
            - str  → Error message if computation fails

    Notes:
        - For binary classification, SHAP values for class index 1 ("positive" class)
          are used consistently.
        - The function includes safeguards to handle differences in SHAP output
          structure across library versions.
        - Errors are caught and returned instead of raising exceptions to prevent
          application crashes in interactive environments like Streamlit.
    """
    
    try:
        # Tree Based models have a different SHAP explainer and output format.
        if model_name in ["Gradient Boosting (Best)", "Random Forest", "Decision Tree"]:
            explainer = shap.TreeExplainer(model)

            # Compute for all test data (global explanations)
            raw_all = explainer.shap_values(X_test)
            # Compute for single input (Local explanations)
            raw_single = explainer.shap_values(input_df)

            base_val = explainer.expected_value

            # Handle different output shapes
            if isinstance(raw_all, list):
                # Old SHAP — list of [class0, class1]
                sv_all    = raw_all[1]
                sv_single = raw_single[1][0]
                base_val  = base_val[1] if hasattr(base_val, '__len__') else base_val
            elif np.array(raw_all).ndim == 3:
                # New SHAP — shape (n_samples, n_features, n_classes)
                sv_all    = np.array(raw_all)[:, :, 1]
                sv_single = np.array(raw_single)[0, :, 1]
                base_val  = base_val[1] if hasattr(base_val, '__len__') else base_val
            else:
                sv_all    = raw_all
                sv_single = raw_single[0]

        # Non tree models use the general SHAP explainer with a single output format
        else: 
            explainer = shap.Explainer(model, X_train)
            # Compute Shap values for all test data and single input
            obj_all    = explainer(X_test)
            obj_single = explainer(input_df)
            # Extracts SHAP values, handling both old and new SHAP output formats
            sv_all    = obj_all.values[:, :, 1] if obj_all.values.ndim == 3 \
                        else obj_all.values
            sv_single = obj_single.values[0, :, 1] if obj_single.values.ndim == 3 \
                        else obj_single.values[0]
            
            base_val  = explainer.expected_value

        return sv_single, sv_all, base_val, True

    except Exception as e:
        return None, None, None, str(e)


# --------------------------------------------------
# Compute SHAP values for the selected model and input
# --------------------------------------------------
# The compute_shap() function returns:
# - sv_single : SHAP values for a single input sample (local explanation)
# - sv_all    : SHAP values for the entire test dataset (global explanation)
# - base_val  : Expected value (baseline prediction of the model)
# - shap_ok   : Status flag → True if successful, otherwise contains error message
sv_single, sv_all, base_val, shap_ok = compute_shap(
    model, selected_model_name, input_df, X_test, X_train, feature_names
)
# Validate SHAP computation
shap_computed = shap_ok is True

# --------------------------------------------------
# Dark chart helper
# --------------------------------------------------
def dark_fig(figsize=(6, 4)):
    """
    Create a dark-themed matplotlib figure for consistent UI styling.

    Parameters:
        figsize (tuple): Size of the figure (width, height)

    Returns:
        fig (Figure): Matplotlib figure object
        ax (Axes): Matplotlib axes object for plotting
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white", labelsize=8)
    ax.spines[:].set_color("#333")
    return fig, ax


# ==================== TABS ====================
tab1, tab2, tab3 = st.tabs(["SHAP", "LIME", "SHAP vs LIME"])


# ── TAB 1: SHAP ──
with tab1:
    st.subheader("SHAP Explanations")

    if not shap_computed:
        st.error(f"Could not compute SHAP values: {shap_ok}")
    else:
        col_g, col_l = st.columns(2)

        with col_g:
            st.markdown("**Global feature importance (test set)**")
            mean_abs = pd.Series(
                np.abs(sv_all).mean(axis=0),
                index=feature_names
            ).sort_values(ascending=True).tail(12)

            fig, ax = dark_fig()
            ax.barh(mean_abs.index, mean_abs.values, color="#3b82f6")
            ax.set_xlabel("Mean |SHAP value|", color="white")
            ax.set_title("Top 12 most important features", color="white", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_l:
            st.markdown(f"**Local explanation — {source_label}**")
            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP":    sv_single
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

        st.markdown("**Feature contributions table**")
        top_feats = shap_df["Feature"].tolist()
        raw_vals  = input_df.iloc[0]
        display_df = pd.DataFrame({
            "Feature":    top_feats,
            "Value":      [round(float(raw_vals[f]), 3) for f in top_feats],
            "SHAP value": [round(sv_single[feature_names.index(f)], 4) for f in top_feats],
            "Direction":  ["→ Pass" if sv_single[feature_names.index(f)] > 0
                           else "→ Fail" for f in top_feats]
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ── TAB 2: LIME ──
with tab2:
    st.subheader("LIME Explanations")
    st.caption("LIME approximates the model locally around a single prediction.")

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
            st.markdown(f"**LIME local explanation — {source_label}**")
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

        with col_b:
            st.markdown("**LIME weights table**")
            st.dataframe(
                lime_df.assign(Weight=lime_df["Weight"].round(4)),
                use_container_width=True, hide_index=True
            )
            st.markdown("---")
            st.markdown("**Prediction probabilities**")
            st.dataframe(pd.DataFrame({
                "Outcome":     ["Fail", "Pass"],
                "Probability": [f"{round(pred_proba[0]*100,1)}%",
                                f"{round(pred_proba[1]*100,1)}%"]
            }), use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"LIME explanation failed: {e}")


# ── TAB 3: SHAP vs LIME ──
with tab3:
    st.subheader("SHAP vs LIME — Side-by-side Comparison")
    st.caption("Both methods applied to the same student. Spearman correlation measures how much they agree on feature rankings.")

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

            st.markdown("---")
            st.markdown("**Agreement between SHAP and LIME**")
            m1, m2, m3 = st.columns(3)
            m1.metric("Spearman correlation", f"{round(corr, 3)}")
            m2.metric("P-value", f"{round(pval, 4)}")
            m3.metric("Agreement level",
                      "Strong" if corr > 0.6 else
                      "Moderate" if corr > 0.3 else "Weak")

            if pval < 0.05:
                st.success(
                    f"Statistically significant agreement (p < 0.05). "
                    f"Both methods {'strongly agree' if corr > 0.6 else 'moderately agree'} "
                    f"on which features matter most."
                )
            else:
                st.warning("Correlation is not statistically significant for this sample.")

            st.markdown("**Feature rank comparison**")
            shap_top10 = shap_series.sort_values(ascending=False).head(10)
            lime_top10 = lime_series.sort_values(ascending=False).head(10)
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