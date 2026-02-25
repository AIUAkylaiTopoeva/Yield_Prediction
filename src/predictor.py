import pickle
import pandas as pd
import os

MODEL_PATH = os.path.join("models", "crop_model.pkl")
COLUMNS_PATH = os.path.join("models", "model_columns.pkl")
STANDARDS_PATH = os.path.join("data", "kg_standards.csv")

# Кэшируем standards (чтобы не читать каждый раз)
_df_standards = None

def load_standards():
    global _df_standards
    if _df_standards is None:
        try:
            _df_standards = pd.read_csv(STANDARDS_PATH)
        except FileNotFoundError:
            raise ValueError("Standards file not found")
    return _df_standards

def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(COLUMNS_PATH, "rb") as f:
            columns = pickle.load(f)
        return model, columns
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        raise ValueError(f"Error loading model: {str(e)}")

def predict_yield(region, crop, n, p, k, rainfall, ph):
    model, columns = load_model()
    input_data = {
        "Region": region,
        "Crop": crop,
        "N": n,
        "P": p,
        "K": k,
        "Rainfall": rainfall,
        "pH": ph
    }
    df = pd.DataFrame([input_data])
    encoded = pd.get_dummies(df).reindex(columns=columns, fill_value=0)
    return round(model.predict(encoded)[0], 2)

def recommend_crop(region, n, p, k, rainfall, ph):
    df_standards = load_standards()
    crops = df_standards[df_standards["Region"] == region]["Crop"].unique()
    yields = {}
    for crop in crops:
        yields[crop] = predict_yield(region, crop, n, p, k, rainfall, ph)
    best_crop = max(yields, key=yields.get)
    return best_crop, yields[best_crop]