import shap
import pickle
import pandas as pd
import os

def get_shap_explanation(region, crop, n, p, k, rainfall, ph):
    with open(os.path.join("models", "crop_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join("models", "model_columns.pkl"), "rb") as f:
        cols = pickle.load(f)

    data = {"Region": region, "Crop": crop, "N": n, "P": p, "K": k, "Rainfall": rainfall, "pH": ph}
    df = pd.DataFrame([data])
    encoded = pd.get_dummies(df).reindex(columns=cols, fill_value=0)

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(encoded)

    imp = pd.Series(values[0], index=cols).abs().sort_values(ascending=False).head(5)
    lines = [f"{feat}: {val:.3f}" for feat, val in imp.items()]
    return "Top influences:\n" + "\n".join(lines)

