import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_lottie import st_lottie
import requests
from datetime import datetime, timedelta

# --- Machine Learning Imports ---
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SURYASHAKTI AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS (Glassmorphism & Modern UI) ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: rgb(15,23,42);
        background: linear-gradient(160deg, rgba(15,23,42,1) 0%, rgba(30,41,59,1) 50%, rgba(15,23,42,1) 100%);
    }
    
    /* Glassmorphism Cards */
    div.css-1r6slb0.e1tzin5v2, div.stMetric {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00E5FF !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 10, 20, 0.9);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Buttons */
    div.stButton > button {
        background: linear-gradient(45deg, #00E5FF, #2979FF);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(0, 229, 255, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION CONSTANTS ---
PANEL_AREA_M2 = 2.0 
GRID_RATE = 7.0     
LOAD_PROFILE = {
    "AC (1.5 Ton)":     {"qty": 1, "kwh": 1.5},   
    "Fans":             {"qty": 5, "kwh": 0.075}, 
    "LEDs":             {"qty": 10, "kwh": 0.01}, 
    "Washing Machine":  {"qty": 1, "kwh": 0.5}    
}

# --- HELPER FUNCTIONS ---
def load_lottieurl(url):
    try:
        r = requests.get(url)
        if r.status_code != 200: return None
        return r.json()
    except: return None

@st.cache_data
def load_data(file):
    try:
        raw_df = pd.read_csv(file)
        raw_df.rename(columns={
            'timestamp': 'datetime',
            'temp_C': 'temperature_C', 
            'precipitation_probability_pct': 'cloud_percentage' 
        }, inplace=True)
        
        df = raw_df[['datetime', 'irradiance_W_m2', 'temperature_C', 'cloud_percentage']].copy()
        try:
            df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M')
        except:
            df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def calculate_physics_generation(row, num_panels):
    irradiance = row['irradiance_W_m2']
    cloud = row['cloud_percentage']
    
    efficiency = 0.0
    if 0 <= cloud <= 30: efficiency = 1.0
    elif 30 < cloud <= 60: efficiency = 0.8
    elif 60 < cloud <= 90: efficiency = 0.6
    else: efficiency = 0.2

    total_area = num_panels * PANEL_AREA_M2
    power_kw = (irradiance * total_area * efficiency) / 1000.0
    return max(0.0, power_kw)

@st.cache_resource
def train_model(df, num_panels):
    training_data = df.copy()
    training_data['hour'] = training_data['datetime'].dt.hour
    
    # Apply physics formula to get ground truth
    training_data['power_output'] = training_data.apply(lambda x: calculate_physics_generation(x, num_panels), axis=1)
    
    features = ['irradiance_W_m2', 'temperature_C', 'cloud_percentage', 'hour']
    X = training_data[features]
    y = training_data['power_output']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    return model, scaler, r2

# --- MAIN APP LOGIC ---

def main():
    # Sidebar Layout
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # File Upload
        uploaded_file = st.file_uploader("Upload Weather CSV", type=['csv'])
        
        st.divider()
        
        # User Config
        num_panels = st.number_input("Solar Panels Qty", min_value=1, value=10, step=1)
        
        st.divider()
        st.markdown("### 🔋 System Status")
        
        lottie_solar = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_z3pnisgt.json")
        if lottie_solar:
            st_lottie(lottie_solar, height=150, key="sidebar_anim")
        
        st.info("Suryashakti AI v2.0 Online")

    # Main Content Area
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            # Initialize Model
            if 'model_trained' not in st.session_state:
                with st.spinner('🧠 Initializing AI Core & Training Neural Mesh...'):
                    model, scaler, accuracy = train_model(df, num_panels)
                    st.session_state['model'] = model
                    st.session_state['scaler'] = scaler
                    st.session_state['accuracy'] = accuracy
                    st.session_state['model_trained'] = True
                    time.sleep(1) # Cinematic delay
            
            model = st.session_state['model']
            scaler = st.session_state['scaler']
            
            # Header
            col1, col2 = st.columns([3, 1])
            with col1:
                st.title("🌞 SURYASHAKTI AI")
                st.markdown(f"#### Advanced Solar Analytics | Model Accuracy: **{st.session_state['accuracy']*100:.2f}%**")
            with col2:
                # Simulation Control: Pick a random time index
                if 'sim_index' not in st.session_state:
                    st.session_state['sim_index'] = np.random.randint(0, len(df) - (24*31))
                
                if st.button("🎲 Randomize Simulation Time"):
                    st.session_state['sim_index'] = np.random.randint(0, len(df) - (24*31))
            
            # Get Current Simulation State
            idx = st.session_state['sim_index']
            current_row = df.iloc[idx]
            current_dt = current_row['datetime']
            
            # --- DASHBOARD TABS ---
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📡 Live Status", 
                "🤖 AI Forecast", 
                "🏠 Load Monitor", 
                "🧠 Smart Scheduler", 
                "💰 3D Comparative ROI"
            ])
            
            # --- TAB 1: LIVE STATUS ---
            with tab1:
                # Predict Current Output
                input_feat = pd.DataFrame([{
                    'irradiance_W_m2': current_row['irradiance_W_m2'],
                    'temperature_C': current_row['temperature_C'],
                    'cloud_percentage': current_row['cloud_percentage'],
                    'hour': current_dt.hour
                }])
                input_scaled = scaler.transform(input_feat)
                current_gen = model.predict(input_scaled).item()
                
                st.markdown("### 📊 Real-Time Telemetry")
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("📅 Date/Time", current_dt.strftime('%d/%m %H:%M'))
                m2.metric("⚡ Current Output", f"{current_gen:.3f} kW", delta="Active")
                m3.metric("☀️ Irradiance", f"{current_row['irradiance_W_m2']} W/m²")
                m4.metric("☁️ Cloud Cover", f"{current_row['cloud_percentage']}%", delta_color="inverse")
                
                # 3D Element: Gauge Chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = current_gen,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "System Load Capacity (kW)"},
                    gauge = {
                        'axis': {'range': [0, (num_panels * PANEL_AREA_M2 * 1.2)/1000 * 1000]}, # Rough max
                        'bar': {'color': "#00E5FF"},
                        'steps': [
                            {'range': [0, 1], 'color': 'rgba(255, 255, 255, 0.1)'},
                            {'range': [1, 5], 'color': 'rgba(255, 255, 255, 0.3)'}
                        ]
                    }
                ))
                fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                st.plotly_chart(fig_gauge, use_container_width=True)

            # --- TAB 2: FORECAST ---
            with tab2:
                st.markdown("### 🔮 24-Hour Generation Forecast")
                
                start = current_dt.replace(hour=0, minute=0)
                end = current_dt.replace(hour=23, minute=59)
                mask = (df['datetime'] >= start) & (df['datetime'] <= end)
                day_data = df.loc[mask].copy()
                day_data['hour'] = day_data['datetime'].dt.hour
                
                features = day_data[['irradiance_W_m2', 'temperature_C', 'cloud_percentage', 'hour']]
                day_data['pred_kw'] = model.predict(scaler.transform(features))
                
                # Interactive Area Chart
                fig = px.area(day_data, x='datetime', y='pred_kw', 
                              title=f"Solar Curve for {start.strftime('%d/%m/%Y')}",
                              labels={'pred_kw': 'Power (kW)', 'datetime': 'Time'},
                              color_discrete_sequence=["#00E5FF"])
                
                fig.add_vline(x=current_dt, line_width=2, line_dash="dash", line_color="red", annotation_text="NOW")
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", 
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white")
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Total Prediction
                st.success(f"📉 Total Energy Expected Today: **{day_data['pred_kw'].sum():.2f} kWh**")

            # --- TAB 3: LOAD MONITOR ---
            with tab3:
                st.markdown("### 🔌 Dynamic Load Balancing")
                
                c1, c2 = st.columns([1, 2])
                
                with c1:
                    user_load = st.slider("Current House Load (kW)", 0.0, 10.0, 1.0, 0.1)
                    net_energy = user_load - current_gen
                    
                    if net_energy > 0:
                        st.error(f"⚠ Grid Import: {net_energy:.2f} kW")
                        st.caption(f"Cost: ₹{net_energy*GRID_RATE:.2f}/hr")
                    else:
                        st.success(f"✅ Export/Charging: {abs(net_energy):.2f} kW")
                        
                with c2:
                    # Visualizing Flow
                    vals = [current_gen, user_load]
                    labs = ["Solar Gen", "Home Load"]
                    cols = ["#00E5FF", "#FF5252"] if net_energy > 0 else ["#00E5FF", "#69F0AE"]
                    
                    fig_pie = go.Figure(data=[go.Pie(labels=labs, values=vals, hole=.6, marker_colors=cols)])
                    fig_pie.update_layout(
                        title_text="Energy Balance", 
                        annotations=[dict(text='NET', x=0.5, y=0.5, font_size=20, showarrow=False)],
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white")
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)

            # --- TAB 4: SCHEDULER ---
            with tab4:
                st.markdown("### 🧠 Intelligent Appliance Scheduler")
                st.markdown("Based on future weather patterns, here are the best times to run heavy loads (Washing Machine, EV Charging).")
                
                end_dt = current_dt + timedelta(hours=24)
                mask = (df['datetime'] > current_dt) & (df['datetime'] <= end_dt)
                future_df = df.loc[mask].copy()
                future_df['hour'] = future_df['datetime'].dt.hour
                
                f_feats = future_df[['irradiance_W_m2', 'temperature_C', 'cloud_percentage', 'hour']]
                future_df['pred'] = model.predict(scaler.transform(f_feats))
                
                best_slots = future_df.sort_values('pred', ascending=False).head(3)
                
                # Card Layout for Recommendations
                cols = st.columns(3)
                for i, (idx, row) in enumerate(best_slots.iterrows()):
                    with cols[i]:
                        st.markdown(f"""
                        <div style="background-color:rgba(0, 229, 255, 0.1); padding:15px; border-radius:10px; border:1px solid #00E5FF; text-align:center;">
                            <h3>#{i+1} Best Slot</h3>
                            <h2>{row['datetime'].strftime('%H:%M')}</h2>
                            <p>Expected Gen: <b>{row['pred']:.2f} kW</b></p>
                            <p>{row['datetime'].strftime('%d/%m')}</p>
                        </div>
                        """, unsafe_allow_html=True)

            # --- TAB 5: 3D COMPARATIVE ANALYSIS ---
            with tab5:
                st.markdown("### 💰 Monthly Financial Projection (30 Days)")
                
                with st.spinner("Running Multi-Scenario Simulation..."):
                    # Calculation Logic (Same as original)
                    end_sim_dt = current_dt + timedelta(days=30)
                    mask = (df['datetime'] >= current_dt) & (df['datetime'] < end_sim_dt)
                    sim_df = df.loc[mask].copy()
                    sim_df['hour'] = sim_df['datetime'].dt.hour
                    
                    sim_feats = sim_df[['irradiance_W_m2', 'temperature_C', 'cloud_percentage', 'hour']]
                    sim_df['solar_gen'] = model.predict(scaler.transform(sim_feats))
                    sim_df['date'] = sim_df['datetime'].dt.date

                    bill_A, bill_B, bill_C = 0, 0, 0
                    
                    for date, day_group in sim_df.groupby('date'):
                        peak_sun_idx = day_group['solar_gen'].idxmax()
                        peak_hour = day_group.loc[peak_sun_idx, 'hour']
                        
                        for _, row in day_group.iterrows():
                            h = row['hour']
                            solar = row['solar_gen']
                            
                            base_load = (LOAD_PROFILE["Fans"]["kwh"] * LOAD_PROFILE["Fans"]["qty"])
                            if h >= 22 or h < 6: base_load += (LOAD_PROFILE["AC (1.5 Ton)"]["kwh"] * LOAD_PROFILE["AC (1.5 Ton)"]["qty"])
                            if 18 <= h <= 23: base_load += (LOAD_PROFILE["LEDs"]["kwh"] * LOAD_PROFILE["LEDs"]["qty"])
                            wm_kwh = LOAD_PROFILE["Washing Machine"]["kwh"]

                            # A: Grid Only
                            load_A = base_load + (wm_kwh if h == 20 else 0)
                            bill_A += (load_A * GRID_RATE)

                            # B: Solar Unoptimized
                            load_B = base_load + (wm_kwh if h == 20 else 0)
                            bill_B += (max(0, load_B - solar) * GRID_RATE)

                            # C: Solar Optimized
                            load_C = base_load + (wm_kwh if h == peak_hour else 0)
                            bill_C += (max(0, load_C - solar) * GRID_RATE)
                    
                    # Results Display
                    st.markdown(f"#### Total Savings Opportunity: **₹{bill_A - bill_C:,.2f}**")
                    
                    # 3D Bar Chart Comparison
                    x_data = ['Grid Only', 'Lazy Solar', 'AI Optimized']
                    y_data = [bill_A, bill_B, bill_C]
                    colors = ['#FF5252', '#FFD740', '#69F0AE']
                    
                    # Create 3D looking Bar chart (Plotly uses 2D bars but we can style them)
                    fig_bar = go.Figure(data=[go.Bar(
                        x=x_data, 
                        y=y_data,
                        marker_color=colors,
                        text=[f"₹{x:,.0f}" for x in y_data],
                        textposition='auto'
                    )])
                    
                    fig_bar.update_layout(
                        title="30-Day Cost Projection",
                        xaxis_title="Scenario",
                        yaxis_title="Cost (₹)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                        bargap=0.5
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # 3D Element: Surface Plot showing Irradiance Landscape
                    st.markdown("#### 🌄 Solar Landscape (Time vs Cloud vs Output)")
                    
                    # Downsample for performance for 3D plot
                    plot_df = sim_df.iloc[::4] # Take every 4th point
                    
                    fig_3d = go.Figure(data=[go.Mesh3d(
                        x=plot_df['hour'],
                        y=plot_df['cloud_percentage'],
                        z=plot_df['solar_gen'],
                        opacity=0.8,
                        color='cyan'
                    )])
                    
                    fig_3d.update_layout(
                        scene = dict(
                            xaxis_title='Hour of Day',
                            yaxis_title='Cloud Cover %',
                            zaxis_title='Power Gen (kW)',
                            bgcolor="rgba(0,0,0,0)"
                        ),
                        margin=dict(l=0, r=0, b=0, t=0),
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)

    else:
        # Landing Page when no file is uploaded
        st.container()
        col1, col2 = st.columns(2)
        with col1:
            st.title("SURYASHAKTI AI")
            st.markdown("### The Future of Home Energy Management")
            st.write("Please upload the `surat_weather_Finalv4_3years.csv` file in the sidebar to initialize the neural network.")
        with col2:
             lottie_welcome = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_ilp96sd6.json")
             if lottie_welcome:
                 st_lottie(lottie_welcome, height=300)

if __name__ == "__main__":
    main()
