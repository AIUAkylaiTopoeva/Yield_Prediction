import pandas as pd
import os

def load_standards():
    path = os.path.join("data", "kg_standards.csv")
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None

def get_recommendation(region, crop, user_n, user_p, user_k, user_ph):
    df = load_standards()
    if df is None:
        return "Error: standards file not found"

    row = df[(df["Region"] == region) & (df["Crop"] == crop)]
    if row.empty:
        return "No data for this region and crop"

    row = row.iloc[0]
    opt_n, opt_p, opt_k, opt_ph = row["Optimal_N"], row["Optimal_P"], row["Optimal_K"], row["Optimal_pH"]

    messages = []

    if user_n < opt_n * 0.8:
        messages.append(f"N deficiency: {opt_n - user_n:.1f} mg/kg below optimal. Yield may drop 20-30%.")
    elif user_n > opt_n * 1.5:
        messages.append("N excess: too high, risk of quality loss. Reduce application.")

    if user_p < opt_p * 0.7:
        messages.append(f"P critical deficiency: {opt_p - user_p:.1f} mg/kg below. Yield may drop 30-40%.")

    if user_k < opt_k * 0.8:
        messages.append("K deficiency: possible quality and yield reduction 10-20%.")

    ph_diff = abs(user_ph - opt_ph)
    if ph_diff > 0.5:
        if user_ph < opt_ph:
            messages.append(f"pH too low ({user_ph}): add lime to raise pH.")
        else:
            messages.append(f"pH too high ({user_ph}): add sulfur to lower pH.")

    if not messages:
        messages.append("Soil parameters are optimal.")

    return "\n".join(messages)