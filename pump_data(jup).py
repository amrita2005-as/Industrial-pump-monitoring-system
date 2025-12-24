import pandas as pd
import numpy as np

# SETTINGS
INPUT_FILE = "1 pump values (15).xlsx"

OUTPUT_4H = "pumps_2year_4hour_historical.xlsx"
OUTPUT_DAILY = "pumps_2year_daily.xlsx"
OUTPUT_MONTHLY = "pumps_2year_monthly.xlsx"

RATED_LPH_PER_KW = 1.5
INTERVAL_MINUTES = 15
TARGET_INTERVAL_HOURS = 4
YEARS_HISTORY = 2
PUMPS = 4

np.random.seed(42)

# LOAD BASE DATA
df = pd.read_excel(INPUT_FILE)

df.columns = [
    "Timestamp",
    "Power_kW",
    "Energy_kWh",
    "Flow_LPH",
    "Flow_Liters",
    "LPH_per_KW",
    "Efficiency_pct"
]

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# BUILD 2 YEARS
intervals_per_day = int(24 * 60 / INTERVAL_MINUTES)
intervals_total = intervals_per_day * 365 * YEARS_HISTORY

df_long = pd.concat(
    [df.sample(frac=1, replace=True)] * int(np.ceil(intervals_total / len(df))),
    ignore_index=True
).iloc[:intervals_total]

df_long["Timestamp"] = pd.date_range(
    end=pd.Timestamp.today().floor("15min"),
    periods=len(df_long),
    freq="15min"
)

# DEGRADATION + NOISE
degradation = np.linspace(1.0, 0.85, len(df_long))
sensor_noise = np.random.normal(1.0, 0.03, len(df_long))

# CORE PARAMETERS
df_long["Power_kW"] *= sensor_noise
df_long["Flow_LPH"] *= sensor_noise * degradation

df_long["Energy_kWh"] = df_long["Power_kW"] * (INTERVAL_MINUTES / 60)
df_long["Flow_Liters"] = df_long["Flow_LPH"] * (INTERVAL_MINUTES / 60)
df_long["LPH_per_KW"] = df_long["Flow_LPH"] / df_long["Power_kW"]
df_long["Efficiency_pct"] = (df_long["LPH_per_KW"] / RATED_LPH_PER_KW) * 100

# NEW PARAMETERS
# Voltage (V): slow drift + noise
df_long["Voltage_V"] = (
    np.random.uniform(10500, 11500) *
    np.random.normal(1.0, 0.02, len(df_long))
).clip(9000, 12500)

# Current (A): correlated with power
df_long["Current_A"] = (
    (df_long["Power_kW"] / df_long["Power_kW"].max()) * 300 +
    np.random.normal(0, 10, len(df_long))
).clip(150, 320)

# Frequency (Hz): almost constant
df_long["Frequency_Hz"] = (
    np.random.normal(49.8, 0.15, len(df_long))
).clip(47.5, 50)

# Temperature (°C): power + degradation driven
df_long["Temperature_C"] = (
    60 +
    (df_long["Power_kW"] / df_long["Power_kW"].max()) * 35 +
    (1 - degradation) * 25 +
    np.random.normal(0, 2, len(df_long))
).clip(60, 115)

# Vibration (mm/s): degradation sensitive
df_long["Vibration_mm_s"] = (
    0.13 +
    (1 - degradation) * 0.15 +
    np.random.normal(0, 0.01, len(df_long))
).clip(0.125, 0.3)

# MULTI-PUMP VARIATION
all_pumps = []

for pump in range(1, PUMPS + 1):
    temp = df_long.copy()
    temp["Pump_ID"] = f"Pump_{pump}"

    bias = np.random.normal(1.0, 0.04)

    temp["Power_kW"] *= bias
    temp["Flow_LPH"] *= bias
    temp["Current_A"] *= bias
    temp["Temperature_C"] *= (1 + (bias - 1) * 0.5)
    temp["Vibration_mm_s"] *= (1 + (bias - 1) * 0.6)

    temp["Energy_kWh"] = temp["Power_kW"] * (INTERVAL_MINUTES / 60)
    temp["Flow_Liters"] = temp["Flow_LPH"] * (INTERVAL_MINUTES / 60)
    temp["LPH_per_KW"] = temp["Flow_LPH"] / temp["Power_kW"]
    temp["Efficiency_pct"] = (temp["LPH_per_KW"] / RATED_LPH_PER_KW) * 100

    all_pumps.append(temp)

df_15min = pd.concat(all_pumps, ignore_index=True)

# 4-HOUR AGGREGATION
df_15min["group_4h"] = (
    df_15min.groupby("Pump_ID").cumcount() //
    int(TARGET_INTERVAL_HOURS * 60 / INTERVAL_MINUTES)
)

agg_cols = {
    "Timestamp": "first",
    "Power_kW": "mean",
    "Flow_LPH": "mean",
    "Voltage_V": "mean",
    "Current_A": "mean",
    "Frequency_Hz": "mean",
    "Temperature_C": "mean",
    "Vibration_mm_s": "mean"
}

df_4h = df_15min.groupby(["Pump_ID", "group_4h"]).agg(agg_cols).reset_index()

df_4h["Energy_kWh"] = df_4h["Power_kW"] * TARGET_INTERVAL_HOURS
df_4h["Flow_Liters"] = df_4h["Flow_LPH"] * TARGET_INTERVAL_HOURS
df_4h["LPH_per_KW"] = df_4h["Flow_LPH"] / df_4h["Power_kW"]
df_4h["Efficiency_pct"] = (df_4h["LPH_per_KW"] / RATED_LPH_PER_KW) * 100

df_4h.to_excel(OUTPUT_4H, index=False)

# DAILY
df_daily = df_4h.copy()
df_daily["Date"] = df_daily["Timestamp"].dt.date

df_daily = df_daily.groupby(["Pump_ID", "Date"]).mean(numeric_only=True).reset_index()
df_daily.to_excel(OUTPUT_DAILY, index=False)

# MONTHLY
df_monthly = df_4h.copy()
df_monthly["Month"] = df_monthly["Timestamp"].dt.to_period("M")

df_monthly = df_monthly.groupby(["Pump_ID", "Month"]).mean(numeric_only=True).reset_index()
df_monthly.to_excel(OUTPUT_MONTHLY, index=False)

print("✅ Historical data with electrical & condition parameters generated successfully")
