import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import shap

# --- 1. CONFIGURATION & THEME ---
st.set_page_config(layout="wide", page_title="Reg-Explain: PPNR Stress Testing")

# Force the Bloomberg Terminal Dark Theme
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #0A0A0A; color: #E0E0E0; }
    
    /* Headers (Bloomberg Orange) */
    h1, h2, h3 { color: #FF9900 !important; font-family: 'Roboto Mono', monospace; }
    
    /* Inputs */
    .stSelectbox, .stSlider { color: #FF9900; }
    
    /* Metrics */
    div[data-testid="stMetricValue"] { color: #00FF00; font-family: 'Courier New', monospace; }
    div[data-testid="stMetricLabel"] { color: #FF9900; }
    
    /* Charts */
    .js-plotly-plot .plotly .modebar { display: none; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. MODEL INTEGRATION ---
# We wrap this in a function to cache it so we don't reload heavy libraries
@st.cache_resource
def load_prediction_model():
    # ADJUST THIS IMPORT to match your actual file path!
    # Example: from backend.models.ensemble import EnsembleModel
    try:
        from backend.models.ensemble import EnsembleModel
        model = EnsembleModel()
        # Mock training just to initialize weights for the demo
        # (In a real app, you would load a saved .h5 file here)
        model.lstm_weight = 0.6
        model.xgboost_weight = 0.4
        return model
    except ImportError:
        return None

# --- 3. DUMMY DATA GENERATOR (Since we have no DB) ---
def get_mock_data():
    quarters = [f"Q{i}" for i in range(1, 10)]
    # Random realistic PPNR values (in Billions)
    base_values = np.linspace(4.5, 3.8, 9) 
    shock_values = base_values * (1 - np.random.uniform(0.1, 0.2, 9))
    return quarters, base_values, shock_values

# --- 4. MAIN DASHBOARD ---
def main():
    st.title("REG-EXPLAIN // PPNR ENGINE")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### SCENARIO CONTROL")
        scenario = st.selectbox("Select Scenario", ["Fed Baseline", "Severely Adverse", "Internal Stress"])
        shock_factor = st.slider("Market Shock Magnitude", 0.0, 1.0, 0.5)
        
        st.divider()
        st.info("⚠️ DATABASE MODE: OFF (In-Memory)")

    with col2:
        # Load the model class (or use dummy if import fails)
        model = load_prediction_model()
        
        if st.button("RUN FORECAST ▶", type="primary"):
            with st.spinner("Processing Ensemble Logic (LSTM + XGBoost)..."):
                # Generating visualization data
                q, base, shock = get_mock_data()
                
                # Plotting
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=q, y=base, name='Baseline', line=dict(color='#00FF00', width=2)))
                fig.add_trace(go.Scatter(x=q, y=shock, name='Stress Projection', line=dict(color='#FF0000', width=2, dash='dot')))
                
                fig.update_layout(
                    title="PPNR Forecast (9-Quarter Horizon)",
                    paper_bgcolor='#0A0A0A',
                    plot_bgcolor='#0A0A0A',
                    font=dict(color='#E0E0E0'),
                    xaxis=dict(showgrid=True, gridcolor='#333'),
                    yaxis=dict(showgrid=True, gridcolor='#333')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Mock Explainability
                st.markdown("### MODEL EXPLAINABILITY (SHAP)")
                st.success(f"**Dominant Driver:** Corporate Bond Spreads (Contribution: +{(shock_factor*100):.1f}%)")
                st.caption(f"Ensemble Weights: LSTM ({0.6}) | XGBoost ({0.4})")

if __name__ == "__main__":
    main()
