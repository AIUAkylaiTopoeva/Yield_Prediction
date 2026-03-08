# AgroExpert Kyrgyzstan — Yield Prediction + DSS

AgroExpert Kyrgyzstan is a hybrid decision-support project for crop yield estimation.

It combines:
- a **Random Forest regression model** trained on synthetic agronomic scenarios derived from regional standards;
- a rule-based **DSS module** that returns actionable recommendations on nutrient balance, pH, and rainfall;
- a **Streamlit app** for interactive analysis.

## What the project does

For a selected **region + crop** and user-entered soil/climate parameters, the app can:
1. predict expected yield (t/ha);
2. compare prediction with regional baseline potential;
3. generate agronomic recommendations;
4. display feature importance and SHAP-based local explanation;
5. suggest the best crop for current conditions.

## Repository structure

```text
.
├── app.py                     # Streamlit UI
├── train_model.py             # Synthetic data generation + model training
├── generate_figure4.py        # Uncertainty plot from forest trees
├── data/
│   ├── kg_standards.csv       # Regional agronomic standards
│   └── raw/Crop_recommendation.csv
├── models/
│   ├── crop_model.pkl
│   └── model_columns.pkl
├── src/
│   ├── predictor.py           # Inference + best crop recommendation
│   ├── dss_logic.py           # Rule-based recommendations
│   ├── interpretability.py    # Feature importance extraction
│   └── shap_utils.py          # SHAP explanation helper
└── figures/
