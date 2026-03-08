 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/README.md b/README.md
index e69de29bb2d1d6434b8b29ae775ad8c2e48c5391..8ed55409dd871b6f4ff30fb791fade2272b1b9fb 100644
--- a/README.md
+++ b/README.md
@@ -0,0 +1,96 @@
+# AgroExpert Kyrgyzstan — Yield Prediction + DSS
+
+AgroExpert Kyrgyzstan is a hybrid decision-support project for crop yield estimation.
+It combines:
+- a **Random Forest regression model** trained on synthetic agronomic scenarios derived from regional standards;
+- a rule-based **DSS module** that returns actionable recommendations on nutrient balance, pH, and rainfall;
+- a **Streamlit app** for interactive analysis.
+
+## What the project does
+
+For a selected **region + crop** and user-entered soil/climate parameters, the app can:
+1. predict expected yield (t/ha);
+2. compare prediction with regional baseline potential;
+3. generate agronomic recommendations;
+4. display feature importance and SHAP-based local explanation;
+5. suggest the best crop for current conditions.
+
+## Repository structure
+
+```text
+.
+├── app.py                     # Streamlit UI
+├── train_model.py             # Synthetic data generation + model training
+├── generate_figure4.py        # Uncertainty plot from forest trees
+├── data/
+│   ├── kg_standards.csv       # Regional agronomic standards
+│   └── raw/Crop_recommendation.csv
+├── models/
+│   ├── crop_model.pkl
+│   └── model_columns.pkl
+├── src/
+│   ├── predictor.py           # Inference + best crop recommendation
+│   ├── dss_logic.py           # Rule-based recommendations
+│   ├── interpretability.py    # Feature importance extraction
+│   └── shap_utils.py          # SHAP explanation helper
+└── figures/
+```
+
+## Installation
+
+```bash
+python -m venv .venv
+source .venv/bin/activate
+pip install -r requirements.txt
+```
+
+## Run the app
+
+```bash
+streamlit run app.py
+```
+
+## (Optional) Retrain the model
+
+```bash
+python train_model.py
+```
+
+This regenerates:
+- `models/crop_model.pkl`
+- `models/model_columns.pkl`
+
+## (Optional) Rebuild uncertainty figure
+
+```bash
+python generate_figure4.py
+```
+
+## Known limitations
+
+1. **Model training data is synthetic** (generated from standards with perturbation), not direct field observations.
+2. **No automated tests** yet.
+3. SHAP explanation is computed per request and may be slow on constrained machines.
+4. No strict input validation layer (beyond Streamlit widgets).
+
+## Priority improvements (recommended)
+
+1. Add automated tests:
+   - unit tests for `predict_yield`, `recommend_crop`, and `get_recommendation`;
+   - smoke test for model and column artifact compatibility.
+2. Add schema checks for `kg_standards.csv` at startup.
+3. Move model loading to cached singleton/resource in app runtime.
+4. Add `.env`-style configurable paths for datasets/models.
+5. Add CI pipeline (`ruff`/`flake8` + `pytest`).
+
+## Minimal test/check commands
+
+```bash
+python -m py_compile app.py src/*.py train_model.py generate_figure4.py
+python train_model.py
+python generate_figure4.py
+```
+
+## License
+
+Specify your license here (for example, MIT).
 
EOF
)
