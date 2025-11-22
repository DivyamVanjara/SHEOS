# app.py
import streamlit as st
import pandas as pd
import numpy as np
import random
import io
from datetime import timedelta

# ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Visualization
import plotly.express as px
import plotly.graph_objects as go
import base64
import time

# -----------------------
# Config / Defaults
# -----------------------
DEFAULT_CSV_PATH = "surat_weather_Finalv4_3years.csv"  # fallback on disk if user doesn't upload
PANEL_AREA_M2_DEFAULT = 2.0
GRID_RATE_DEFAULT = 7.0

LOAD_PROFILE = {
    "AC (1.5 Ton)":    {"qty": 1, "kwh": 1.5},
    "Fans":            {"qty": 5, "kwh": 0.075},
    "LEDs":            {"qty": 10, "kwh": 0.01},
    "Washing Machine": {"qty": 1, "kwh": 0.5}
}

st.set_page_config(page_title="Suryashakti AI", page_icon="🌞", layout="wide")

# ---- Styling / header ----
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg, #0f172a 0%, #021124 60%); color: #e6eef8; }
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border-radius: 14px; padding: 18px; box-shadow: 0 8px 24px rgba(2,6,23,0.6); }
    .small { font-size: 0.9rem; color: #cfe8ff; }
    .muted { color: #9fb8d6; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# Header row
col1, col2 = st.columns([3,1])
with col1:
    st.markdown("# 🌞 SURYASHAKTI AI — Solar Home Dashboard")
    st.markdown("<div class='muted'>AI-driven solar generation forecasting, scheduling and ROI simulations — interactive, modern and production-ready UI.</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div style='text-align:right'><small class='muted'>Built for Streamlit • 3D + Plotly • RandomForest</small></div>", unsafe_allow_html=True)

st.markdown("")  # spacer

# -----------------------
# Sidebar: Inputs / Upload
# -----------------------
with st.sidebar:
    st.markdown("## Settings & Upload", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload weather CSV (timestamp, irradiance_W_m2, temp_C, precipitation_probability_pct)", type=["csv"])
    use_default = st.checkbox("Use default CSV on disk (if exists)", value=True)
    panel_count = st.slider("Number of solar panels", min_value=1, max_value=100, value=6)
    panel_area_m2 = st.number_input("Panel area (m² per panel)", value=PANEL_AREA_M2_DEFAULT, step=0.1)
    grid_rate = st.number_input("Grid rate (₹ per kWh)", value=GRID_RATE_DEFAULT, step=0.5)
    st.markdown("---")
    if st.button("(Re)train / load model"):
        st.session_state.get("trigger_retrain", False)
        st.session_state["trigger_retrain"] = not st.session_state.get("trigger_retrain", False)
    st.markdown("## About")
    st.markdown("This app trains a RandomForest on a 3-year dataset. It uses a physics-based label (irradiance → power) and then learns to predict power from features. All five options from the original script are provided via the main controls.")

# -----------------------
# Utility functions & cached model
# -----------------------
def _parse_and_prep(df):
    # normalize names, parse datetime
    df = df.copy()
    df = df.rename(columns={
        'timestamp': 'datetime',
        'temp_C': 'temperature_C',
        'temperature_C': 'temperature_C',
        'precipitation_probability_pct': 'cloud_percentage',
        'cloud_percentage': 'cloud_percentage'
    })
    if 'datetime' in df.columns:
        try:
            df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M')
        except:
            df['datetime'] = pd.to_datetime(df['datetime'])
    else:
        st.error("CSV must contain a 'datetime' column.")
        raise RuntimeError("Missing datetime")

    # Ensure numeric columns exist
    for col in ['irradiance_W_m2', 'temperature_C', 'cloud_percentage']:
        if col not in df.columns:
            st.error(f"CSV missing required column: {col}")
            raise RuntimeError("Missing column")
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    df = df.sort_values('datetime').reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def load_dataframe(uploaded_file_obj, use_default_flag):
    # returns dataframe or raises
    if uploaded_file_obj is not None:
        df = pd.read_csv(uploaded_file_obj)
        return _parse_and_prep(df)
    else:
        if use_default_flag:
            try:
                df = pd.read_csv(DEFAULT_CSV_PATH)
                return _parse_and_prep(df)
            except FileNotFoundError:
                st.warning("Default CSV not found on disk. Please upload a file.")
                return None
        else:
            st.warning("Please upload CSV or enable default CSV.")
            return None

@st.cache_data(show_spinner=False)
def generate_physics_label(df, num_panels, panel_area_m2):
    df2 = df.copy()
    # calculate 'hour'
    df2['hour'] = df2['datetime'].dt.hour
    def calc(row):
        irradiance = row['irradiance_W_m2']
        cloud = row['cloud_percentage']
        if 0 <= cloud <= 30: efficiency = 1.0
        elif 30 < cloud <= 60: efficiency = 0.8
        elif 60 < cloud <= 90: efficiency = 0.6
        else: efficiency = 0.2
        total_area = num_panels * panel_area_m2
        power_kw = (irradiance * total_area * efficiency) / 1000.0
        return max(0.0, power_kw)
    df2['power_output'] = df2.apply(calc, axis=1)
    return df2

@st.cache_resource(show_spinner=False)
def train_model(df_with_labels):
    # features & target
    X = df_with_labels[['irradiance_W_m2', 'temperature_C', 'cloud_percentage', 'hour']].copy()
    y = df_with_labels['power_output'].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return model, scaler, r2

def predict_with_model(model, scaler, features_df):
    X = features_df[['irradiance_W_m2', 'temperature_C', 'cloud_percentage', 'hour']].copy()
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    return preds

# -----------------------
# Load data and train
# -----------------------
df = load_dataframe(uploaded_file, use_default)

if df is None:
    st.stop()

# compute labels using physics formula (depends on panel_count & panel_area)
with st.spinner("Preparing labels & (re)training model if needed..."):
    df_labeled = generate_physics_label(df, panel_count, panel_area_m2)

    # Add a small guard to ensure we have enough rows
    if len(df_labeled) < 100:
        st.error("Not enough data rows for training.")
        st.stop()

    # Retain training keyed by hash of data + panel_count to retrain when changed
    # Using cache_resource will keep model in memory between reruns
    model, scaler, r2 = train_model(df_labeled)

st.success(f"Model ready — R²: {r2*100:.2f}%")

# store in session for main interactions
st.session_state["df"] = df_labeled
st.session_state["model"] = model
st.session_state["scaler"] = scaler
st.session_state["panel_count"] = panel_count
st.session_state["panel_area_m2"] = panel_area_m2
st.session_state["grid_rate"] = grid_rate

# -----------------------
# Main UI: Tabs for options
# -----------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Live Status", "AI Forecast", "Load Monitor", "Scheduler", "ROI Comparison"])

# Utility to pick a random index (simulate current time from dataset)
def pick_current_index(df):
    max_start = len(df) - (24 * 31) - 1
    if max_start <= 0:
        return 0
    return random.randint(0, max_start)

if "current_index" not in st.session_state:
    st.session_state["current_index"] = pick_current_index(df_labeled)

current_row = st.session_state["df"].iloc[st.session_state["current_index"]]

# ----- Tab 1: Live Status -----
with tab1:
    st.markdown("### 📡 Live Generation Status")
    colA, colB = st.columns([3,1])
    with colA:
        st.metric("Date & Time", current_row['datetime'].strftime("%d/%m/%Y %H:%M"))
        st.metric("Temperature (°C)", f"{current_row['temperature_C']:.1f}")
        st.metric("Clouds (%)", f"{current_row['cloud_percentage']:.0f}")
        st.metric("Irradiance (W/m²)", f"{current_row['irradiance_W_m2']:.1f}")

        # compute live prediction
        input_df = pd.DataFrame([{
            'irradiance_W_m2': current_row['irradiance_W_m2'],
            'temperature_C': current_row['temperature_C'],
            'cloud_percentage': current_row['cloud_percentage'],
            'hour': current_row['datetime'].hour
        }])
        gen_kw = predict_with_model(model, scaler, input_df).item()
        st.metric("Estimated Output (kW)", f"{gen_kw:.3f}", delta=f"{panel_count} panels")

    with colB:
        # 3D ornament (three.js) embedded for style
        html_three = """
        <div id="canvas" style="width:100%;height:220px;"></div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r134/three.min.js"></script>
        <script>
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(35, 1.5, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({antialias:true, alpha:true});
        renderer.setSize(320,220);
        document.getElementById('canvas').appendChild(renderer.domElement);
        const light = new THREE.DirectionalLight(0xffffff, 1);
        light.position.set(5,10,7);
        scene.add(light);
        const geom = new THREE.BoxGeometry(2,0.05,1);
        const mat = new THREE.MeshStandardMaterial({color:0x0fb4ff, metalness:0.3, roughness:0.4});
        const panel = new THREE.Mesh(geom, mat);
        panel.rotation.x = -0.2;
        scene.add(panel);
        camera.position.z = 4;
        function animate(){ requestAnimationFrame(animate); panel.rotation.y += 0.01; renderer.render(scene, camera); }
        animate();
        </script>
        """
        st.components.v1.html(html_three, height=240)

    st.markdown("---")
    st.button("Refresh current timestamp (simulate new now)", on_click=lambda: st.session_state.update({"current_index": pick_current_index(st.session_state["df"])}))

# ----- Tab 2: AI Forecast -----
with tab2:
    st.markdown("### 🤖 AI Day Forecast")
    st.write("Shows hourly predicted generation for the selected date (the current row's date).")

    forecast_date = current_row['datetime'].date()
    df = st.session_state["df"]
    start = pd.to_datetime(pd.to_datetime(forecast_date))
    mask = (df['datetime'].dt.date == forecast_date)
    day_df = df.loc[mask].copy()
    day_df['hour'] = day_df['datetime'].dt.hour
    if day_df.empty:
        st.warning("No data for this date. Try refreshing current timestamp.")
    else:
        day_df['pred_kw'] = predict_with_model(model, scaler, day_df[['irradiance_W_m2','temperature_C','cloud_percentage','hour']])
        # line chart
        fig = px.line(day_df, x='hour', y='pred_kw', markers=True, title=f"Predicted generation — {forecast_date}")
        fig.update_layout(yaxis_title="Power (kW)", xaxis_title="Hour of day")
        st.plotly_chart(fig, use_container_width=True)

        # interactive table
        st.dataframe(day_df[['datetime','irradiance_W_m2','temperature_C','cloud_percentage','pred_kw']].rename(columns={
            'datetime':'Time','irradiance_W_m2':'Irradiance','temperature_C':'Temp (°C)','cloud_percentage':'Clouds (%)','pred_kw':'Pred kW'
        }), height=300)

        # 3D field: scatter of 24 hours mapped onto a small grid and height = pred_kw
        fig3d = go.Figure(data=[go.Scatter3d(
            x=np.cos(2*np.pi*day_df['hour']/24),
            y=np.sin(2*np.pi*day_df['hour']/24),
            z=day_df['pred_kw'],
            mode='markers+lines',
            marker=dict(size=6),
            line=dict(color='orange')
        )])
        fig3d.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Pred kW'), height=450)
        st.plotly_chart(fig3d, use_container_width=True)

# ----- Tab 3: Load Monitor -----
with tab3:
    st.markdown("### 🏠 Dynamic Load Monitor")
    st.markdown("Enter a current household load and see net import/export & cost impact.")
    load_input = st.number_input("Enter current house load (kW)", value=1.0, step=0.1)
    input_df = pd.DataFrame([{
        'irradiance_W_m2': current_row['irradiance_W_m2'],
        'temperature_C': current_row['temperature_C'],
        'cloud_percentage': current_row['cloud_percentage'],
        'hour': current_row['datetime'].hour
    }])
    gen = predict_with_model(model, scaler, input_df).item()
    net = load_input - gen
    if net > 0:
        st.error(f"⚠ Grid Import: {net:.3f} kW — Cost: ₹{net * grid_rate:.2f} / hr")
    else:
        st.success(f"✅ Grid Export / Charging: {abs(net):.3f} kW")

    # small gauge style chart
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=gen,
        delta={'reference': load_input, 'position':'bottom'},
        gauge={'axis': {'range':[0, max(load_input*1.5, gen*1.5, 5)]}},
        title={'text': "Estimated Generation (kW)"}
    ))
    st.plotly_chart(fig_g, use_container_width=True)

# ----- Tab 4: Scheduler -----
with tab4:
    st.markdown("### 🧠 Smart Scheduler — Best hours to run heavy appliances")
    df_future = df.copy()
    current_dt = current_row['datetime']
    end_dt = current_dt + timedelta(hours=24)
    mask = (df_future['datetime'] > current_dt) & (df_future['datetime'] <= end_dt)
    future_df = df_future.loc[mask].copy()
    if future_df.empty:
        st.warning("No future records in next 24h — try refreshing current index.")
    else:
        future_df['hour'] = future_df['datetime'].dt.hour
        future_df['pred'] = predict_with_model(model, scaler, future_df[['irradiance_W_m2','temperature_C','cloud_percentage','hour']])
        best_slots = future_df.sort_values('pred', ascending=False).head(3)
        st.write("Best times to run heavy appliances (top 3):")
        for _, row in best_slots.iterrows():
            st.write(f"🕒 {row['datetime'].strftime('%d/%m %H:%M')} — Pred: {row['pred']:.2f} kW")
        # Bar chart of next 24 hours
        fig_bar = px.bar(future_df, x='datetime', y='pred', labels={'pred':'Pred kW','datetime':'Time'}, title="Next 24h predicted generation")
        st.plotly_chart(fig_bar, use_container_width=True)

# ----- Tab 5: Comparative ROI -----
with tab5:
    st.markdown("### 💰 Monthly Financial Savings (30 days simulation)")
    st.write("Compares three scenarios over the next 30 days based on current index as start.")
    current_dt = current_row['datetime']
    end_dt = current_dt + timedelta(days=30)
    mask = (df['datetime'] >= current_dt) & (df['datetime'] < end_dt)
    sim_df = df.loc[mask].copy()
    if sim_df.empty:
        st.warning("Not enough data to run 30-day simulation from the selected current index.")
    else:
        sim_df['hour'] = sim_df['datetime'].dt.hour
        sim_df['solar_gen'] = predict_with_model(model, scaler, sim_df[['irradiance_W_m2','temperature_C','cloud_percentage','hour']])
        bill_A = 0.0
        bill_B = 0.0
        bill_C = 0.0

        sim_df['date'] = sim_df['datetime'].dt.date
        for date, day_group in sim_df.groupby('date'):
            peak_idx = day_group['solar_gen'].idxmax()
            peak_hour = day_group.loc[peak_idx, 'hour']
            for _, row in day_group.iterrows():
                h = row['hour']
                solar = row['solar_gen']
                hourly_load_base = 0.0
                hourly_load_base += LOAD_PROFILE["Fans"]["kwh"] * LOAD_PROFILE["Fans"]["qty"]
                if h >= 22 or h < 6:
                    hourly_load_base += LOAD_PROFILE["AC (1.5 Ton)"]["kwh"] * LOAD_PROFILE["AC (1.5 Ton)"]["qty"]
                if 18 <= h <= 23:
                    hourly_load_base += LOAD_PROFILE["LEDs"]["kwh"] * LOAD_PROFILE["LEDs"]["qty"]
                wm_kwh = LOAD_PROFILE["Washing Machine"]["kwh"]
                # Scenario A: Grid only
                load_A = hourly_load_base
                if h == 20: load_A += wm_kwh
                bill_A += load_A * grid_rate
                # Scenario B: Solar but runs WM at 20:00 (bad timing)
                load_B = hourly_load_base
                if h == 20: load_B += wm_kwh
                net_B = max(0, load_B - solar)
                bill_B += net_B * grid_rate
                # Scenario C: Solar optimized — run WM at peak hour
                load_C = hourly_load_base
                if h == peak_hour: load_C += wm_kwh
                net_C = max(0, load_C - solar)
                bill_C += net_C * grid_rate

        # display
        colA, colB = st.columns(2)
        with colA:
            st.metric("Grid Only (30d)", f"₹{bill_A:,.2f}")
            st.metric("Solar (Bad Timing)", f"₹{bill_B:,.2f}")
            st.metric("Solar + AI (Optimized)", f"₹{bill_C:,.2f}")
        with colB:
            savings_vs_grid = bill_A - bill_C
            savings_vs_unopt = bill_B - bill_C
            st.metric("Savings vs Grid (30d)", f"₹{savings_vs_grid:,.2f}")
            st.metric("Extra Savings vs Lazy Solar", f"₹{savings_vs_unopt:,.2f}")

        # donut chart
        fig_pie = go.Figure(data=[go.Pie(labels=['Grid Only','Solar Bad Timing','Solar Optimized'],
                                         values=[bill_A, bill_B, bill_C], hole=0.45)])
        fig_pie.update_layout(title="30-day cost comparison")
        st.plotly_chart(fig_pie, use_container_width=True)

        # offer CSV download summary
        results = pd.DataFrame({
            "Scenario": ["Grid Only", "Solar Bad Timing", "Solar Optimized"],
            "Cost_30d_Rs": [bill_A, bill_B, bill_C]
        })
        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("Download summary CSV", data=csv, file_name="suryashakti_summary_30d.csv", mime="text/csv")

# -----------------------
# Footer / tips
# -----------------------
st.markdown("---")
st.markdown("**Tips:** Use the side panel to upload your own CSV. Increase number of panels or panel area in settings to see differences. The 3D ornament is decorative — primary 3D analytics use the Plotly 3D chart for interpretability.")
