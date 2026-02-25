import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

MODEL_PATH = os.path.join("models", "crop_model.pkl")
COLUMNS_PATH = os.path.join("models", "model_columns.pkl")

# 1️⃣ Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(COLUMNS_PATH, "rb") as f:
    columns = pickle.load(f)

# 2️⃣ Generate synthetic test data
np.random.seed(42)

n_samples = 1000

data = {
    "Region": np.random.choice(["Chui", "Osh", "Naryn"], n_samples),
    "Crop": np.random.choice(["Wheat", "Barley", "Corn"], n_samples),
    "N": np.random.uniform(20, 150, n_samples),
    "P": np.random.uniform(10, 80, n_samples),
    "K": np.random.uniform(30, 200, n_samples),
    "Rainfall": np.random.uniform(200, 800, n_samples),
    "pH": np.random.uniform(5.5, 8.0, n_samples),
}

df = pd.DataFrame(data)
encoded = pd.get_dummies(df).reindex(columns=columns, fill_value=0)

# 3️⃣ Get predictions from all trees
all_tree_preds = np.array([
    tree.predict(encoded)
    for tree in model.estimators_
])

# 4️⃣ Calculate uncertainty (std across trees)
uncertainty = np.std(all_tree_preds, axis=0)

# 5️⃣ Plot histogram
plt.figure()
plt.hist(uncertainty, bins=30)
plt.xlabel("Prediction Uncertainty (t/ha)")
plt.ylabel("Frequency")
plt.title("Uncertainty Distribution Across Predictions")
plt.tight_layout()

# 6️⃣ Save figure
plt.savefig("fig4_uncertainty.png", dpi=300)
plt.close()

print("Figure 4 saved as fig4_uncertainty.png")