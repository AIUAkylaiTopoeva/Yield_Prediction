import pickle
import pandas as pd
import os

def get_feature_importance():
    try:
        with open(os.path.join("models", "crop_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join("models", "model_columns.pkl"), "rb") as f:
            cols = pickle.load(f)
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        return pd.DataFrame({"error": [f"Error loading model: {str(e)}"]})  # Возвращаем DF с ошибкой для Streamlit

    imp = model.feature_importances_
    df = pd.DataFrame({"feature": cols, "importance": imp})
    return df.sort_values("importance", ascending=False)