import streamlit as st
import pandas as pd
from src.predictor import predict_yield, recommend_crop
from src.dss_logic import get_recommendation
from src.interpretability import get_feature_importance
from src.shap import get_shap_explanation
import os

st.set_page_config(page_title="AgroExpert KG", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #f5f7f5; }
h1, h2, h3 { color: #2e7d32; }
.metric-box {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    border-left: 6px solid #2e7d32;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

standards_path = os.path.join("data", "kg_standards.csv")
df_standards = pd.read_csv(standards_path)
df_standards.replace("No expert opinion", pd.NA, inplace=True)

regions = sorted(df_standards["Region"].unique())

st.title("AgroExpert Kyrgyzstan")
st.markdown("Hybrid **Machine Learning + Decision Support System** for crop yield prediction and recommendations.")
st.markdown("---")

st.sidebar.header("Input Parameters")

region = st.sidebar.selectbox("Region", regions)
available_crops = df_standards[df_standards["Region"] == region]["Crop"].dropna().unique()
crop = st.sidebar.selectbox("Crop", sorted(available_crops))

st.sidebar.subheader("Soil Analysis (mg/kg)")
n = st.sidebar.slider("Nitrogen (N)", 0, 150, 60)
p = st.sidebar.slider("Phosphorus (P)", 0, 100, 35)
k = st.sidebar.slider("Potassium (K)", 0, 450, 220)
ph = st.sidebar.slider("pH", 5.0, 8.0, 6.5, step=0.1)

st.sidebar.subheader("Climate")
rainfall = st.sidebar.number_input("Average Rainfall (mm)", value=450)

if st.sidebar.button("Run Analysis"):
    try:
        row = df_standards[(df_standards["Region"] == region) & (df_standards["Crop"] == crop)].iloc[0]
        base_yield = row["Expected_Yield_ha"]

        pred_yield = predict_yield(region, crop, n, p, k, rainfall, ph)
        advice = get_recommendation(region, crop, n, p, k, ph)

        st.header(f"{crop} in {region}")
        col1, col2 = st.columns(2)
        col1.metric("Predicted Yield", f"{pred_yield} t/ha")
        col2.metric("Regional Potential", f"{base_yield} t/ha")

        st.subheader("Recommendations")
        if "Deficiency" in advice or "Critical" in advice:
            st.error(advice)
        elif "Excess" in advice:
            st.warning(advice)
        else:
            st.success(advice)

        st.subheader("Feature Importance")
        st.dataframe(get_feature_importance().head(10))

        st.subheader("SHAP Explanation")
        st.text(get_shap_explanation(region, crop, n, p, k, rainfall, ph))

    except Exception as e:
        st.error(f"Error: {str(e)}")

if st.sidebar.button("Recommend Best Crop"):
    try:
        best_crop, best_yield = recommend_crop(region, n, p, k, rainfall, ph)
        st.success(f"Best crop: **{best_crop}** → {best_yield} t/ha")
    except Exception as e:
        st.error(f"Error: {str(e)}")