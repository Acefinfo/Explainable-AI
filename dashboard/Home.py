import streamlit as st

st.set_page_config(
    page_title="XAI - Student Perfomance ",
    page_icon = "",
    layout= "wide",
    initial_sidebar_state= "expanded"
)

st.title("Explanable AI - Student Performance ")
st.markdown(
    "This system predicts whether a student will **pass or fail** "
    "and explains every prediction using **SHAP** and **LIME**."
)

st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Dataset", "395 student samples")
with col2:
    st.metric("Models", "5 trained models")
with col3:
    st.metric("Best Model", "Gradient Boosting", "F1-Score: 95.1%")

st.markdown("---")
st.markdown(
    """
    **Use the sidebar to navigate:**
    - **Predict** — enter student details and get a prediction
    - **Explanations** — explore SHAP and LIME explanations
    - **Model Comparison** — compare all 5 models
    - **About** — project details
    """
)

st.caption("BSc (Hons) Computing 2025/26 · Aashutosh Thapa · c7466915 · leeds beckett university" )