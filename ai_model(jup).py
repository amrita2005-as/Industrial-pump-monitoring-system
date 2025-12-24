import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import shap
import os

# -----------------------------
# PARAMETERS
# -----------------------------
HIST_FILE = "pumps_2year_monthly.xlsx"
OUTPUT_PRED = "predicted_pumps_all_params.xlsx"
SHAP_FILE = "shap_values.xlsx"

NEW_PARAMS = ["Voltage_V", "Current_A", "Frequency_Hz", "Temperature_C", "Vibrations_mm_s"]
OLD_PARAMS = ["Power_kW", "Energy_kWh", "Flow_LPH", "LPH_per_KW", "Efficiency_pct"]

ALL_PARAMS = OLD_PARAMS + NEW_PARAMS
FUTURE_MONTHS = 24  # Predict 24 months ahead

# -----------------------------
# LOAD HISTORICAL DATA
# -----------------------------
df_hist = pd.read_excel(HIST_FILE)
df_hist["Month"] = pd.to_datetime(df_hist["Month"])

# Ensure new parameters exist and are numeric
for param in NEW_PARAMS:
    if param not in df_hist.columns:
        # Generate synthetic historical values if missing
        if param == "Voltage_V":
            df_hist[param] = np.random.uniform(9000, 12500, size=len(df_hist))
        elif param == "Current_A":
            df_hist[param] = np.random.uniform(150, 320, size=len(df_hist))
        elif param == "Frequency_Hz":
            df_hist[param] = np.random.uniform(47.5, 50, size=len(df_hist))
        elif param == "Temperature_C":
            df_hist[param] = np.random.uniform(60, 115, size=len(df_hist))
        elif param == "Vibrations_mm_s":
            df_hist[param] = np.random.uniform(0.125, 0.3, size=len(df_hist))

# -----------------------------
# PREDICTIONS
# -----------------------------
predictions = []
shap_values_dict = {}

for pump in df_hist["Pump_ID"].unique():
    df_p = df_hist[df_hist["Pump_ID"] == pump].copy()
    df_pred_pump = pd.DataFrame()
    df_pred_pump["Pump_ID"] = [pump]*FUTURE_MONTHS
    
    # Future months
    last_month = df_p["Month"].max()
    df_pred_pump["Month"] = pd.date_range(start=last_month + pd.DateOffset(months=1), periods=FUTURE_MONTHS, freq='MS')
    
    for target in ALL_PARAMS:
        # Use all other parameters as features + time index
        features = [col for col in ALL_PARAMS if col != target]
        X = df_p[features].copy()
        X["time_idx"] = np.arange(len(df_p))
        y = df_p[target].values
        
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        model.fit(X, y)
        
        # Prepare future X
        X_future = pd.DataFrame()
        for f in features:
            # Use last known value + small random noise
            X_future[f] = df_p[f].iloc[-1] * (1 + np.random.normal(0, 0.02, FUTURE_MONTHS))
        X_future["time_idx"] = np.arange(len(df_p), len(df_p)+FUTURE_MONTHS)
        
        df_pred_pump[target] = model.predict(X_future)
        
        # SHAP explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_future)
        shap_values_dict[f"{pump}_{target}"] = shap_values.mean(axis=0)
    
    predictions.append(df_pred_pump)

df_pred_all = pd.concat(predictions, ignore_index=True)
df_pred_all.to_excel(OUTPUT_PRED, index=False)
print(f"Predicted data saved to {OUTPUT_PRED}")

# -----------------------------
# SAVE SHAP EXPLANATIONS
# -----------------------------
shap_df = pd.DataFrame(shap_values_dict)
shap_df.to_excel(SHAP_FILE, index=False)
print(f"SHAP explanations saved to {SHAP_FILE}")

# -----------------------------
# GENERATE GRAPHS (NEW PARAMETERS)
# -----------------------------
os.makedirs("new_param_graphs", exist_ok=True)

for pump in df_hist["Pump_ID"].unique():
    df_hist_p = df_hist[df_hist["Pump_ID"] == pump]
    df_pred_p = df_pred_all[df_pred_all["Pump_ID"] == pump]
    
    for param in NEW_PARAMS:
        plt.figure(figsize=(10,4))
        plt.plot(df_hist_p["Month"], df_hist_p[param], marker='o', label="Historical")
        plt.plot(df_pred_p["Month"], df_pred_p[param], marker='x', linestyle='--', label="Predicted")
        plt.title(f"{param} over Time for {pump}")
        plt.xlabel("Month")
        plt.ylabel(param)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"new_param_graphs/{pump}_{param}.png")
        plt.close()

print("Graphs for new parameters generated successfully!")
