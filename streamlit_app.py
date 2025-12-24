# =============================
# IMPORTS
# =============================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from io import BytesIO
from pathlib import Path

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Pump Performance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

# =============================
# CUSTOM CSS - FULLY ADAPTIVE THEME
# =============================
st.markdown("""
<style>
    /* Remove all text color hardcoding - let Streamlit handle it */
    .dashboard-title,
    .dashboard-subtitle,
    .section-header,
    .graph-title,
    .comparison-title,
    .factor-title,
    .stat-value,
    .control-label,
    .download-header,
    .comparison-card,
    .factor-card,
    .info-box,
    .recommendation-box,
    .recommendation-item,
    .factor-item,
    .stat-label,
    .legend-text,
    .recommendation-title,
    .anomaly-title,
    .factor-item strong,
    .recommendation-item strong,
    .anomaly-card p,
    .info-box strong,
    .pump-stats strong {
        color: inherit !important;
    }
    
    /* Main container */
    .main {
        background-color: transparent;
    }
    
    /* Header styling - transparent with borders */
    .dashboard-header {
        background: rgba(128, 128, 128, 0.1);
        padding: 2.5rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        border-left: 5px solid #3498db;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .dashboard-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        letter-spacing: 1px;
    }
    
    .dashboard-subtitle {
        font-size: 1rem;
        margin-top: 0.5rem;
        text-align: center;
        font-weight: 400;
        opacity: 0.8;
    }
    
    /* Section headers */
    .section-header {
        background: rgba(52, 152, 219, 0.1);
        padding: 1rem 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    /* Graph titles */
    .graph-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
        padding-left: 0.5rem;
        border-left: 3px solid #3498db;
        text-align: center;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(52, 152, 219, 0.1);
        border-left: 4px solid #3498db;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Recommendation box */
    .recommendation-box {
        background: rgba(46, 204, 113, 0.1);
        border: 2px solid #27ae60;
        border-left: 5px solid #27ae60;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    .recommendation-title {
        color: #27ae60 !important;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
    }
    
    .recommendation-item {
        font-size: 1rem;
        line-height: 1.6;
        padding: 0.3rem 0;
    }
    
    .optimal-value {
        background: rgba(46, 204, 113, 0.25);
        padding: 2px 10px;
        border-radius: 3px;
        font-weight: 700;
        font-family: 'Courier New', monospace;
        color: #27ae60 !important;
        border: 1px solid rgba(46, 204, 113, 0.3);
    }
    
    /* Comparison section */
    .comparison-card {
        background: rgba(128, 128, 128, 0.05);
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .comparison-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .pump-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .stat-item {
        background: rgba(52, 152, 219, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #3498db;
    }
    
    .stat-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
        opacity: 0.7;
    }
    
    .stat-value {
        font-size: 1.3rem;
        font-weight: 700;
    }
    
    /* Anomaly card */
    .anomaly-card {
        background: rgba(231, 76, 60, 0.1);
        border: 2px solid #e74c3c;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 5px solid #e74c3c;
    }
    
    .anomaly-title {
        color: #e74c3c !important;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
    }
    
    /* Factor analysis */
    .factor-card {
        background: rgba(128, 128, 128, 0.05);
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 4px solid #3498db;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .factor-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .factor-item {
        font-size: 0.95rem;
        line-height: 1.6;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        background: rgba(52, 152, 219, 0.05);
        border-left: 3px solid #95a5a6;
    }
    
    /* Control panel */
    .control-label {
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    
    /* Download section */
    .download-header {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Pump diagram labels - always use contrasting colors */
    .pump-label {
        position: absolute;
        padding: 10px 15px;
        border-radius: 8px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
        z-index: 10;
    }
    
    .pump-label:hover {
        transform: scale(1.08);
        box-shadow: 0 6px 16px rgba(0,0,0,0.5);
    }
    
    .label-title {
        font-size: 12px;
        font-weight: bold;
        margin-bottom: 3px;
        color: white !important;
        letter-spacing: 0.5px;
    }
    
    .label-value {
        font-size: 16px;
        font-weight: bold;
        color: white !important;
    }
    
    .legend-box {
        margin-top: 2rem;
        padding: 1rem;
        background: rgba(128, 128, 128, 0.1);
        border-radius: 8px;
        border-left: 4px solid #3498db;
        text-align: center;
    }
    
    .legend-text {
        margin: 0;
        font-size: 0.9rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    df_h = pd.read_excel("pumps_2year_monthly_updated.xlsx")
    df_f = pd.read_excel("predicted_pumps_all_params.xlsx")
    df_shap = pd.read_excel("shap_values.xlsx")

    for df in [df_h, df_f]:
        df["Month"] = pd.to_datetime(df["Month"])

        if "Flow_LPH" not in df.columns and "Flow_Liters" in df.columns:
            df["Flow_LPH"] = df["Flow_Liters"] / (30 * 24)

        if "Flow_Liters" not in df.columns and "Flow_LPH" in df.columns:
            df["Flow_Liters"] = df["Flow_LPH"] * 30 * 24

    return df_h, df_f, df_shap

df_h, df_f, df_shap = load_data()

# =============================
# ANOMALY DETECTION FUNCTION
# =============================
def detect_anomalies(df, column, method='iqr', threshold=1.5):
    """Detect anomalies using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    anomalies = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return anomalies, lower_bound, upper_bound

def analyze_anomalies(df, params=['Efficiency_pct', 'Power_kW', 'Flow_LPH']):
    """Comprehensive anomaly analysis - per pump"""
    all_anomalies = []
    anomaly_summary = {}
    
    pumps = df['Pump_ID'].unique()
    
    for pump in pumps:
        pump_df = df[df['Pump_ID'] == pump]
        
        for param in params:
            if param in pump_df.columns:
                anomalies, lower, upper = detect_anomalies(pump_df, param)
                
                if len(anomalies) > 0:
                    key = f"{pump}_{param}"
                    anomaly_summary[key] = {
                        'pump': pump,
                        'param': param,
                        'count': len(anomalies),
                        'lower_bound': lower,
                        'upper_bound': upper,
                        'anomalies': anomalies
                    }
                    all_anomalies.extend(anomalies.index.tolist())
    
    return anomaly_summary, len(set(all_anomalies))

# =============================
# OPTIMAL OPERATING CONDITIONS
# =============================
def calculate_optimal_conditions(df, pump_id):
    """Calculate optimal operating conditions for a pump"""
    pump_data = df[df['Pump_ID'] == pump_id]
    
    top_efficiency_threshold = pump_data['Efficiency_pct'].quantile(0.90)
    optimal_data = pump_data[pump_data['Efficiency_pct'] >= top_efficiency_threshold]
    
    optimal_conditions = {}
    params = ['Power_kW', 'Flow_LPH', 'Efficiency_pct']
    
    optional_params = ['Voltage_V', 'Current_A', 'Temp_C', 'Vibrations_mm_s', 'Frequency_Hz']
    for param in optional_params:
        if param in pump_data.columns:
            params.append(param)
    
    for param in params:
        if param in optimal_data.columns:
            optimal_conditions[param] = {
                'mean': optimal_data[param].mean(),
                'min': optimal_data[param].min(),
                'max': optimal_data[param].max(),
                'std': optimal_data[param].std()
            }
    
    return optimal_conditions

# =============================
# DETECT THEME FOR PLOTLY
# =============================
def get_plotly_template():
    """Detect current theme and return appropriate plotly template"""
    # Try to detect Streamlit theme (this is a workaround as direct detection isn't available)
    # Default to dark, but we'll make graphs adaptive anyway
    return "plotly_dark"

# =============================
# HEADER
# =============================
st.markdown("""
<div class="dashboard-header">
    <h1 class="dashboard-title">INDUSTRIAL PUMP PERFORMANCE MONITORING SYSTEM</h1>
    <p class="dashboard-subtitle">Predictive Analytics & Real-Time Performance Tracking</p>
</div>
""", unsafe_allow_html=True)

# =============================
# CONTROLS
# =============================
st.markdown('<div class="section-header">CONTROL PANEL</div>', unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<p class="control-label">SELECT PUMP UNITS</p>', unsafe_allow_html=True)
    pump_ids = sorted(df_h["Pump_ID"].unique())
    selected_pumps = st.multiselect(
        "Select Pump(s)",
        pump_ids,
        default=pump_ids[:1],
        label_visibility="collapsed"
    )

with col2:
    st.markdown('<p class="control-label">TIME RANGE SELECTION</p>', unsafe_allow_html=True)
    min_m = df_h["Month"].min()
    max_m = df_f["Month"].max()
    
    date_range = pd.date_range(min_m, max_m, freq="MS")
    date_options = [d.strftime("%b %Y") for d in date_range]
    
    selected_range = st.select_slider(
        "Select Time Range (Month-Year)",
        options=date_options,
        value=(date_options[0], date_options[-1]),
        label_visibility="collapsed"
    )
    
    start_m = pd.to_datetime(selected_range[0], format="%b %Y")
    end_m = pd.to_datetime(selected_range[1], format="%b %Y")

df_h_sel = df_h[(df_h["Pump_ID"].isin(selected_pumps)) & 
                (df_h["Month"] >= start_m) & 
                (df_h["Month"] <= end_m)]
df_f_sel = df_f[(df_f["Pump_ID"].isin(selected_pumps)) & 
                (df_f["Month"] >= start_m) & 
                (df_f["Month"] <= end_m)]

# =============================
# KPI SECTION WITH PUMP VISUALIZATION
# =============================
time_range_text = f"{start_m.strftime('%b %Y')} to {end_m.strftime('%b %Y')}"
st.markdown(f'<div class="section-header">PERFORMANCE METRICS & SYSTEM OVERVIEW ({len(selected_pumps)} Pump{"s" if len(selected_pumps) > 1 else ""} Selected | {time_range_text}) </div>', unsafe_allow_html=True)

df_combined = pd.concat([df_h_sel, df_f_sel]).sort_values('Month')

def safe_mean(df, column, default=0):
    """Safely calculate mean, handling NaN and missing columns"""
    if column in df.columns:
        valid_data = df[column].dropna()
        if len(valid_data) > 0:
            return valid_data.mean()
    return default

avg_voltage = safe_mean(df_combined, 'Voltage_V')
avg_current = safe_mean(df_combined, 'Current_A')
avg_frequency = safe_mean(df_combined, 'Frequency_Hz')
avg_temp = safe_mean(df_combined, 'Temperature_C')
avg_vibrations = safe_mean(df_combined, 'Vibrations_mm_s')
avg_power = safe_mean(df_combined, 'Power_kW')
avg_efficiency = safe_mean(df_combined, 'Efficiency_pct')
avg_flow = safe_mean(df_combined, 'Flow_LPH')

kpi_spacer1, kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_spacer2 = st.columns([0.3, 2, 2, 2, 2, 0.7])
with kpi_col1:
    st.metric("PEAK EFFICIENCY", f"{df_combined['Efficiency_pct'].max():.2f}%", delta="Combined")
with kpi_col2:
    st.metric("MIN EFFICIENCY", f"{df_combined['Efficiency_pct'].min():.2f}%", delta="Combined")
with kpi_col3:
    total_energy = safe_mean(df_combined, 'Energy_kWh', 0) * len(df_combined)
    st.metric("TOTAL ENERGY", f"{total_energy/1000:.2f} MWh")
with kpi_col4:
    total_flow = safe_mean(df_combined, 'Flow_Liters', 0) * len(df_combined)
    st.metric("TOTAL FLOW", f"{total_flow/1e6:.2f} ML")

st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)

# Centered pump diagram
pump_html = f"""
<style>
    .pump-container {{
        position: relative;
        max-width: 1200px;
        margin: 0 auto;
        background: transparent;
        padding: 2rem;
        padding-bottom: 4rem;
    }}
    
    .pump-label {{
        position: absolute;
        padding: 10px 15px;
        border-radius: 8px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        transition: all 0.3s ease;
        z-index: 10;
    }}
    
    .pump-label:hover {{
        transform: scale(1.08);
        box-shadow: 0 6px 16px rgba(0,0,0,0.5);
    }}
    
    .label-title {{
        font-size: 12px;
        font-weight: bold;
        margin-bottom: 3px;
        color: white;
        letter-spacing: 0.5px;
    }}
    
    .label-value {{
        font-size: 16px;
        font-weight: bold;
        color: white;
    }}
    
    .curved-arrow {{
        position: absolute;
        z-index: 5;
        stroke-width: 2.5;
        fill: none;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
    }}
    
    .pump-image-container {{
        position: relative;
        width: 100%;
        max-width: 900px;
        margin: 0 auto;
        padding-top: 60%;
        background: transparent;
        border-radius: 12px;
    }}
    
    .pump-image {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 90%;
        height: 90%;
        object-fit: contain;
        filter: none;
        background: transparent;
    }}
    
    .legend-box {{
        margin-top: 2rem;
        padding: 1rem;
        background: var(--card-bg);
        border-radius: 8px;
        border-left: 4px solid var(--accent-color);
        text-align: center;
    }}
    
    .legend-text {{
        margin: 0;
        font-size: 0.9rem;
        color: var(--text-primary);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }}
</style>

<div class="pump-container">
    <div class="pump-image-container">
        <!-- SVG for curved arrows -->
        <svg style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;" viewBox="0 0 900 540">
            <!-- Input arrows (curved from left) -->
            <path class="curved-arrow" d="M 80 60 Q 120 60, 180 120" stroke="#27ae60" marker-end="url(#arrow-green)"/>
            <path class="curved-arrow" d="M 80 160 Q 120 160, 180 180" stroke="#27ae60" marker-end="url(#arrow-green)"/>
            <path class="curved-arrow" d="M 80 480 Q 120 450, 180 400" stroke="#27ae60" marker-end="url(#arrow-green)"/>
            
            <!-- Power arrow -->
            <path class="curved-arrow" d="M 290 60 Q 300 90, 280 150" stroke="#f39c12" marker-end="url(#arrow-orange)"/>
            
            <!-- Monitoring arrows (from top and bottom) -->
            <path class="curved-arrow" d="M 450 60 Q 450 100, 450 150" stroke="#e74c3c" marker-end="url(#arrow-red)"/>
            <path class="curved-arrow" d="M 450 480 Q 450 440, 450 390" stroke="#e74c3c" marker-end="url(#arrow-red)"/>
            
            <!-- Output arrows (curved from right) -->
            <path class="curved-arrow" d="M 820 220 Q 780 220, 720 240" stroke="#3498db" marker-end="url(#arrow-blue)"/>
            <path class="curved-arrow" d="M 820 320 Q 780 320, 720 300" stroke="#9b59b6" marker-end="url(#arrow-purple)"/>
            
            <!-- Arrow markers -->
            <defs>
                <marker id="arrow-green" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                    <path d="M0,0 L0,6 L9,3 z" fill="#27ae60" />
                </marker>
                <marker id="arrow-orange" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                    <path d="M0,0 L0,6 L9,3 z" fill="#f39c12" />
                </marker>
                <marker id="arrow-red" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                    <path d="M0,0 L0,6 L9,3 z" fill="#e74c3c" />
                </marker>
                <marker id="arrow-blue" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                    <path d="M0,0 L0,6 L9,3 z" fill="#3498db" />
                </marker>
                <marker id="arrow-purple" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
                    <path d="M0,0 L0,6 L9,3 z" fill="#9b59b6" />
                </marker>
            </defs>
        </svg>
        
        <img src="https://i.ibb.co/KxmbVr7W/pump.png" 
             class="pump-image"
             alt="Pump Diagram">
        
        <!-- INPUT PARAMETERS -->
        <div class="pump-label" style='top: 8%; left: 2%; background: rgba(39, 174, 96, 0.95); border: 2px solid #27ae60;'>
            <div class="label-title">⚡ VOLTAGE</div>
            <div class="label-value">{avg_voltage:.1f} V</div>
        </div>
        
        <div class="pump-label" style='top: 27%; left: 2%; background: rgba(39, 174, 96, 0.95); border: 2px solid #27ae60;'>
            <div class="label-title">⚡ CURRENT</div>
            <div class="label-value">{avg_current:.1f} A</div>
        </div>
        
        <div class="pump-label" style='bottom: 8%; left: 2%; background: rgba(39, 174, 96, 0.95); border: 2px solid #27ae60;'>
            <div class="label-title">⚡ FREQUENCY</div>
            <div class="label-value">{avg_frequency:.1f} Hz</div>
        </div>
        
        <!-- POWER -->
        <div class="pump-label" style='top: 8%; left: 27%; background: rgba(243, 156, 18, 0.95); border: 2px solid #f39c12;'>
            <div class="label-title">⚙️ POWER</div>
            <div class="label-value">{avg_power:.1f} kW</div>
        </div>
        
        <!-- MONITORING PARAMETERS -->
        <div class="pump-label" style='top: 8%; left: 47%; background: rgba(231, 76, 60, 0.95); border: 2px solid #e74c3c;'>
            <div class="label-title">🌡️ TEMPERATURE</div>
            <div class="label-value">{avg_temp:.1f} °C</div>
        </div>
        
        <div class="pump-label" style='bottom: 8%; left: 47%; background: rgba(231, 76, 60, 0.95); border: 2px solid #e74c3c;'>
            <div class="label-title">📊 VIBRATIONS</div>
            <div class="label-value">{avg_vibrations:.2f} mm/s</div>
        </div>
        
        <!-- OUTPUT PARAMETERS -->
        <div class="pump-label" style='top: 38%; right: 2%; background: rgba(52, 152, 219, 0.95); border: 2px solid #3498db;'>
            <div class="label-title">💧 FLOW RATE</div>
            <div class="label-value">{avg_flow:.0f} L/h</div>
        </div>
        
        <div class="pump-label" style='top: 56%; right: 2%; background: rgba(155, 89, 182, 0.95); border: 2px solid #9b59b6;'>
            <div class="label-title">⚡ EFFICIENCY</div>
            <div class="label-value">{avg_efficiency:.1f} %</div>
        </div>
    </div>
    
    <div class="legend-box">
        <p class="legend-text" style="color: inherit;">
            <strong>Legend:</strong> 
            <span style='color: #27ae60; font-weight: bold;'>● Input</span> | 
            <span style='color: #e74c3c; font-weight: bold;'>● Monitoring</span> | 
            <span style='color: #f39c12; font-weight: bold;'>● Power</span> | 
            <span style='color: #3498db; font-weight: bold;'>● Output</span> | 
            <span style='color: #9b59b6; font-weight: bold;'>● Performance</span>
            <br>
            <span style='font-style: italic; font-size: 0.85rem; margin-top: 0.5rem; display: inline-block; opacity: 0.8;'>
                Averaging across {len(selected_pumps)} pump(s) for time range: {time_range_text}
            </span>
        </p>
    </div>
</div>
"""
st.components.v1.html(pump_html, height=780)

# =============================
# PUMP COMPARISON (IF MULTIPLE PUMPS)
# =============================
if len(selected_pumps) > 1:
    st.markdown('<div class="section-header">PUMP COMPARISON ANALYSIS</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="comparison-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="comparison-title">{" VS ".join(selected_pumps)}</div>', unsafe_allow_html=True)
    
    cols = st.columns(len(selected_pumps))
    
    pump_data_list = []
    for idx, pump in enumerate(selected_pumps):
        pump_data = df_f_sel[df_f_sel['Pump_ID'] == pump]
        pump_data_list.append(pump_data)
        
        with cols[idx]:
            st.markdown(f"**{pump} Performance:**")
            st.markdown(f"""
            <div class="pump-stats">
                <div class="stat-item">
                    <div class="stat-label">Avg Efficiency</div>
                    <div class="stat-value">{pump_data['Efficiency_pct'].mean():.2f}%</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Avg Power</div>
                    <div class="stat-value">{pump_data['Power_kW'].mean():.2f} kW</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Avg Flow</div>
                    <div class="stat-value">{pump_data['Flow_LPH'].mean():.0f} L/h</div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Total Energy</div>
                    <div class="stat-value">{pump_data['Energy_kWh'].sum()/1000:.2f} MWh</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    efficiencies = [(pump, df_f_sel[df_f_sel['Pump_ID'] == pump]['Efficiency_pct'].mean()) for pump in selected_pumps]
    powers = [(pump, df_f_sel[df_f_sel['Pump_ID'] == pump]['Power_kW'].mean()) for pump in selected_pumps]
    energies = [(pump, df_f_sel[df_f_sel['Pump_ID'] == pump]['Energy_kWh'].sum()) for pump in selected_pumps]
    
    best_efficiency = max(efficiencies, key=lambda x: x[1])
    worst_efficiency = min(efficiencies, key=lambda x: x[1])
    lowest_power = min(powers, key=lambda x: x[1])
    highest_power = max(powers, key=lambda x: x[1])
    
    st.markdown(f"""
    <div class="info-box">
        <strong>📊 COMPARISON INSIGHTS:</strong><br>
        • <strong>Best Efficiency:</strong> {best_efficiency[0]} at {best_efficiency[1]:.2f}% (vs. {worst_efficiency[0]} at {worst_efficiency[1]:.2f}%)<br>
        • <strong>Most Energy Efficient:</strong> {lowest_power[0]} at {lowest_power[1]:.2f} kW average power consumption<br>
        • <strong>Highest Power Consumption:</strong> {highest_power[0]} at {highest_power[1]:.2f} kW<br>
        • <strong>Efficiency Range:</strong> {best_efficiency[1] - worst_efficiency[1]:.2f}% difference across all pumps<br>
        • <strong>Power Range:</strong> {highest_power[1] - lowest_power[1]:.2f} kW difference across all pumps
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# OPTIMAL OPERATING CONDITIONS
# =============================
st.markdown('<div class="section-header">OPTIMAL OPERATING CONDITIONS</div>', unsafe_allow_html=True)

for pump in selected_pumps:
    optimal_conditions = calculate_optimal_conditions(df_f_sel, pump)
    
    st.markdown(f"""
    <div class="recommendation-box">
        <div class="recommendation-title">⚙️ {pump} - RECOMMENDED OPERATING PARAMETERS</div>
        <p style="margin: 0 0 0.5rem 0; font-style: italic; font-size: 0.9rem;">Based on top 10% efficiency performance data</p>
    """, unsafe_allow_html=True)
    
    param_names = {
        'Efficiency_pct': ('efficiency', '%'),
        'Power_kW': ('power consumption', 'kW'),
        'Flow_LPH': ('flow rate', 'L/h'),
        'Voltage_V': ('voltage', 'V'),
        'Current_A': ('current', 'A'),
        'Temp_C': ('operating temperature', '°C'),
        'Vibrations_mm_s': ('vibration levels', 'mm/s'),
        'Frequency_Hz': ('frequency', 'Hz')
    }
    
    narrative_parts = []
    if 'Efficiency_pct' in optimal_conditions:
        eff = optimal_conditions['Efficiency_pct']
        narrative_parts.append(
            f"<strong>{pump}</strong> achieves peak efficiency of <span class='optimal-value'>{eff['mean']:.2f}%</span> "
            f"(range: {eff['min']:.2f}%-{eff['max']:.2f}%) under optimal conditions."
        )
    
    other_params = []
    for param, values in optimal_conditions.items():
        if param != 'Efficiency_pct' and param in param_names:
            name, unit = param_names[param]
            other_params.append(
                f"{name} at <span class='optimal-value'>{values['mean']:.2f} {unit}</span> "
                f"({values['min']:.2f}-{values['max']:.2f} {unit})"
            )
    
    if len(other_params) > 0:
        if len(other_params) == 1:
            param_text = other_params[0]
        elif len(other_params) == 2:
            param_text = f"{other_params[0]} and {other_params[1]}"
        else:
            param_text = ", ".join(other_params[:-1]) + f", and {other_params[-1]}"
        
        narrative_parts.append(
            f" To maintain this performance, keep {param_text}."
        )
    
    narrative_parts.append(
        f" These settings optimize efficiency, reduce energy costs, and extend equipment lifespan."
    )
    
    narrative = "<div class='recommendation-item'>" + " ".join(narrative_parts) + "</div>"
    
    st.markdown(narrative, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# GRAPHS - ADAPTIVE TO THEME
# =============================
st.markdown('<div class="section-header">OPERATIONAL TRENDS & ANALYSIS</div>', unsafe_allow_html=True)

st.markdown('<div class="info-box"><strong>DATA LEGEND:</strong> Historical measurements (solid) | AI-predicted values (lighter shades with diamond markers)</div>', unsafe_allow_html=True)

colors_historical = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
colors_predicted = ['#7FB7D9', '#D48FB0', '#F8C078', '#E28B7C', '#A3C585']

def plot_param(param, label, unit=""):
    fig = go.Figure()
    
    best_values = {}
    
    for idx, p in enumerate(selected_pumps):
        h = df_h_sel[df_h_sel["Pump_ID"] == p]
        f = df_f_sel[df_f_sel["Pump_ID"] == p]
        
        color_hist = colors_historical[idx % len(colors_historical)]
        color_pred = colors_predicted[idx % len(colors_predicted)]

        fig.add_trace(go.Scatter(
            x=h["Month"], 
            y=h[param],
            name=f"{p} (Historical)",
            mode="lines+markers",
            line=dict(color=color_hist, width=2.5),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=f["Month"], 
            y=f[param],
            name=f"{p} (Predicted)",
            mode="lines+markers",
            line=dict(color=color_pred, width=2.5),
            marker=dict(size=6, symbol='diamond')
        ))
        
        if len(f) > 0:
            if param == 'Efficiency_pct':
                best_idx = f[param].idxmax()
                best_row = f.loc[best_idx]
                best_values[p] = {
                    'value': best_row[param],
                    'month': best_row['Month'],
                    'color': color_pred
                }
            elif param == 'Power_kW':
                best_idx = f[param].idxmin()
                best_row = f.loc[best_idx]
                best_values[p] = {
                    'value': best_row[param],
                    'month': best_row['Month'],
                    'color': color_pred
                }
            elif param == 'Flow_LPH':
                best_idx = f[param].idxmax()
                best_row = f.loc[best_idx]
                best_values[p] = {
                    'value': best_row[param],
                    'month': best_row['Month'],
                    'color': color_pred
                }
    
    first_best = True
    for pump, best in best_values.items():
        fig.add_trace(go.Scatter(
            x=[best['month']],
            y=[best['value']],
            mode='markers',
            marker=dict(
                size=15,
                color='gold',
                symbol='star',
                line=dict(color='white', width=2)
            ),
            name="★ Best Value" if first_best else "",
            showlegend=first_best,
            legendgroup="best",
            hovertemplate=f"<b>{pump} - BEST VALUE</b><br>" +
                         f"Date: %{{x|%b %Y}}<br>" +
                         f"Value: %{{y:.2f}}<extra></extra>"
        ))
        first_best = False

    # Adaptive colors - transparent backgrounds work for both themes
    fig.update_layout(
        xaxis_title="TIMELINE",
        yaxis_title=f"{label} {unit}".upper(),
        template="plotly_white",  # Changed to white template for better theme adaptation
        height=450,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent
        font=dict(size=11, family='Segoe UI, sans-serif'),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False
        )
    )
    
    return fig

c1, c2 = st.columns(2)
with c1:
    st.markdown('<p class="graph-title">PUMP EFFICIENCY PERFORMANCE</p>', unsafe_allow_html=True)
    st.plotly_chart(plot_param("Efficiency_pct", "Efficiency", "(%)"), use_container_width=True)
with c2:
    st.markdown('<p class="graph-title">POWER CONSUMPTION ANALYSIS</p>', unsafe_allow_html=True)
    st.plotly_chart(plot_param("Power_kW", "Power", "(kW)"), use_container_width=True)

st.markdown('<p class="graph-title">FLOW RATE MONITORING</p>', unsafe_allow_html=True)
st.plotly_chart(plot_param("Flow_LPH", "Flow Rate", "(L/h)"), use_container_width=True)

# =============================
# SHAP EXPLAINABILITY
# =============================
st.markdown('<div class="section-header">PERFORMANCE FACTOR ANALYSIS</div>', unsafe_allow_html=True)
for pump in selected_pumps:
    pump_cols = [c for c in df_shap.columns if c.startswith(f"{pump}_")]
    if not pump_cols:
        st.info(f"No factor analysis data available for {pump}.")
        continue

    if 'Month' in df_shap.columns:
        df_shap['Month'] = pd.to_datetime(df_shap['Month'])
        shap_filtered = df_shap[(df_shap['Month'] >= start_m) & (df_shap['Month'] <= end_m)]
        if len(shap_filtered) == 0:
            st.warning(f"No factor analysis data available for {pump} in the selected time range.")
            continue
        shap_vals = shap_filtered[pump_cols].mean()
    else:
        shap_vals = df_shap[pump_cols].mean()
    
    top_factors = shap_vals.abs().sort_values(ascending=False).head(5)

    st.markdown(f'<div class="factor-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="factor-title">{pump} - KEY PERFORMANCE DRIVERS ({time_range_text})</div>', unsafe_allow_html=True)
    
    avg_impact = shap_vals.abs().mean()
    
    factor_texts = []
    for factor in top_factors.index:
        param_name = factor.replace(f"{pump}_", "")
        param_display = param_name.replace("_", " ").title()
        shap_value = shap_vals[factor]
        
        impact_level = "high" if abs(shap_value) > avg_impact * 1.5 else "moderate" if abs(shap_value) > avg_impact else "low"
        
        if shap_value > 0:
            direction = "increases"
            effect = "improvement"
        else:
            direction = "decreases"
            effect = "reduction"
        
        factor_texts.append(f"<strong>{param_display}</strong> (impact: {abs(shap_value):.4f}, {impact_level}) {direction} efficiency with {effect} in performance")
    
    explanation = f"""
    <div class="factor-item">
        Analysis shows that key factors influencing {pump} performance include: {', '.join(factor_texts[:-1])}, and {factor_texts[-1]}. 
        These parameters collectively determine operational efficiency and require continuous monitoring for optimal performance.
    </div>
    """
    st.markdown(explanation, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# DOWNLOAD SECTION
# =============================
st.markdown('<div class="section-header">DATA EXPORT</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="download-header">▼ DATASET DOWNLOADS</p>', unsafe_allow_html=True)
    buffer_h = BytesIO()
    df_h_sel.to_excel(buffer_h, index=False)
    st.download_button(
        "▼ HISTORICAL DATASET", 
        buffer_h.getvalue(), 
        "historical_data.xlsx",
        use_container_width=True
    )
    
    buffer_f = BytesIO()
    df_f_sel.to_excel(buffer_f, index=False)
    st.download_button(
        "▼ PREDICTED DATASET", 
        buffer_f.getvalue(), 
        "predicted_data.xlsx",
        use_container_width=True
    )

with col2:
    st.markdown('<p class="download-header">▼ PARAMETER GRAPH IMAGES</p>', unsafe_allow_html=True)
    
    available_params = ["Current_A", "Frequency_Hz", "Temperature_C", "Vibrations_mm_s", "Voltage_V"]
    graphs_dir = Path("new_param_graphs")
    
    if graphs_dir.exists():
        for pump in selected_pumps:
            st.markdown(f"<strong style='color: #3498db;'>{pump} GRAPHS:</strong>", unsafe_allow_html=True)
            for param in available_params:
                graph_file = graphs_dir / f"{pump}_{param}.png"
                if graph_file.exists():
                    with open(graph_file, "rb") as file:
                        st.download_button(
                            f"▼ {pump} - {param.replace('_', ' ').title()}",
                            data=file.read(),
                            file_name=graph_file.name,
                            mime="image/png",
                            use_container_width=True
                        )
            st.markdown("---")
    else:
        st.info("Parameter graphs folder 'new_param_graphs' not found")

# =============================
# ANOMALY DETECTION & ANALYSIS
# =============================
st.markdown('<div class="section-header">ANOMALY DETECTION & ANALYSIS</div>', unsafe_allow_html=True)

anomaly_summary, total_anomaly_count = analyze_anomalies(df_f_sel)

st.metric("⚠ ANOMALIES DETECTED", f"{total_anomaly_count}", 
          delta="Alert" if total_anomaly_count > 0 else "Normal",
          delta_color="inverse" if total_anomaly_count > 0 else "normal")

if total_anomaly_count > 0:
    for key, data in anomaly_summary.items():
        pump = data['pump']
        param = data['param']
        count = data['count']
        
        with st.expander(f"⚠ {pump} - {param.replace('_', ' ').upper()} - {count} ANOMALIES DETECTED"):
            st.markdown(f"""
            <div class="anomaly-card">
                <div class="anomaly-title">ABNORMAL {param.replace('_', ' ').upper()} DETECTED IN {pump}</div>
                <p><strong>Anomaly Count:</strong> {count} data points</p>
                <p><strong>Normal Operating Range:</strong> {data['lower_bound']:.2f} - {data['upper_bound']:.2f}</p>
                <p><strong>Action Required:</strong> Investigate timestamps below for maintenance review</p>
            </div>
            """, unsafe_allow_html=True)
            
            anomaly_df = data['anomalies'][['Pump_ID', 'Month', param]].copy()
            anomaly_df['Deviation'] = anomaly_df[param].apply(
                lambda x: f"{((x - data['upper_bound'])/data['upper_bound']*100):.1f}% above" 
                if x > data['upper_bound'] 
                else f"{((data['lower_bound'] - x)/data['lower_bound']*100):.1f}% below"
            )
            st.dataframe(anomaly_df, use_container_width=True)
else:
    st.success("✓ No anomalies detected. All parameters are within normal operating ranges.")

# =============================
# FOOTER
# =============================
st.markdown("---")
st.markdown("""
<div style='text-align: center; opacity: 0.7; padding: 1rem; font-family: monospace;'>
    <p><strong>INDUSTRIAL PUMP PERFORMANCE MONITORING SYSTEM</strong></p>
    <p style='font-size: 0.85rem;'>AI-Powered Predictive Maintenance | Real-Time Analytics | Performance Optimization</p>
</div>
""", unsafe_allow_html=True)