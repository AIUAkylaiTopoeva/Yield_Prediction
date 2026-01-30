import pickle
import pandas as pd
import os

def get_feature_importance():
    with open(os.path.join("models", "crop_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join("models", "model_columns.pkl"), "rb") as f:
        cols = pickle.load(f)

    imp = model.feature_importances_
    df = pd.DataFrame({"feature": cols, "importance": imp})
    return df.sort_values("importance", ascending=False)