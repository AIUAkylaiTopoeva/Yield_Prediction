import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

file_path = os.path.join("data", "kg_standards.csv")
df = pd.read_csv(file_path)
df.replace(["", " ", "No expert opinion"], np.nan, inplace=True)

numeric = ["Optimal_N", "Optimal_P", "Optimal_K", "Optimal_pH", "Expected_Yield_ha"]
for col in numeric:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric)

rows = []
for _ in range(5000):
    r = df.sample(1).iloc[0]
    on, op, ok, oph, base = float(r["Optimal_N"]), float(r["Optimal_P"]), float(r["Optimal_K"]), float(r["Optimal_pH"]), float(r["Expected_Yield_ha"])

    n = max(0, on + np.random.uniform(-30, 30))
    p = max(0, op + np.random.uniform(-20, 20))
    k = max(0, ok + np.random.uniform(-50, 50))
    ph = max(5.0, min(8.0, oph + np.random.uniform(-0.7, 0.7)))
    rain = max(100, 450 + np.random.uniform(-150, 150))

    factor = 1.0

    if n < on:
        factor -= 0.3 * (1 - n / on)
    elif n > on * 1.5:
        factor -= 0.1 * (n / on - 1.5)

    if p < op:
        factor -= 0.4 * (1 - p / op)

    if k < ok:
        factor -= 0.2 * (1 - k / ok)

    phd = abs(ph - oph)
    if phd > 0.5:
        factor -= 0.15 * phd

    if rain < 300 or rain > 700:
        factor -= 0.15 * (abs(rain - 450) / 150)

    factor = max(0.3, factor)
    y = max(0, base * factor)

    rows.append([r["Region"], r["Crop"], n, p, k, rain, ph, y])

train_df = pd.DataFrame(rows, columns=["Region", "Crop", "N", "P", "K", "Rainfall", "pH", "Yield"])

X = pd.get_dummies(train_df.drop("Yield", axis=1))
y = train_df["Yield"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("MAE:", round(mean_absolute_error(y_test, pred), 2))
print("R²:", round(r2_score(y_test, pred), 3))

os.makedirs("models", exist_ok=True)
with open("models/crop_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/model_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

print("Model saved")