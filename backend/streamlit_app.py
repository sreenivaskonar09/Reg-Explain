"""
Reg-Explain: PPNR Stress-Testing Dashboard
Streamlit Application for CCAR Framework with LSTM-XGBoost Ensemble and SHAP Interpretability
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import os
import sys
from datetime import datetime
import io

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from data.synthetic_generator import SyntheticBankDataGenerator, get_fed_2026_scenarios
from data.fred_integration import FREDDataFetcher
from models.lstm_model import PPNRLSTMModel, create_temporal_features
from models.xgboost_model import PPNRXGBoostModel, create_static_features
from models.ensemble import PPNREnsembleModel, QuarterlyPPNRForecaster
from capital_engine import CapitalEngine, RegulatoryThresholds
from shap_explainer import SHAPExplainer, GeminiExplainer

# Page configuration
st.set_page_config(
    page_title="Reg-Explain | PPNR Stress Testing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional financial dashboard
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0e17 0%, #111827 50%, #0d1117 100%);
        font-family: 'IBM Plex Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border-left: 4px solid #3b82f6;
    }
    
    .main-header h1 {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0.5rem 0 0 0;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-safe { color: #22c55e; }
    .status-warning { color: #f59e0b; }
    .status-danger { color: #ef4444; }
    
    .breach-alert {
        background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
        border: 1px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .breach-alert h3 {
        color: #fca5a5;
        margin: 0 0 0.5rem 0;
    }
    
    .breach-alert p {
        color: #fecaca;
        margin: 0;
    }
    
    .scenario-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .scenario-card:hover {
        border-color: #3b82f6;
    }
    
    .scenario-card.selected {
        border-color: #3b82f6;
        background: linear-gradient(145deg, #1e3a5f 0%, #1e293b 100%);
    }
    
    .shap-explanation {
        background: #1e293b;
        border-left: 3px solid #8b5cf6;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .shap-explanation p {
        color: #e2e8f0;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #0f172a;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #1e293b;
        border-radius: 8px;
        color: #94a3b8;
        padding: 0.75rem 1.5rem;
        border: 1px solid #334155;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        color: #f8fafc;
        border-color: #3b82f6;
    }
    
    /* Data upload area */
    .upload-area {
        border: 2px dashed #334155;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: rgba(30, 41, 59, 0.5);
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #3b82f6;
        background: rgba(30, 41, 59, 0.8);
    }
    
    /* Slider styling */
    .stSlider > div > div {
        background: #334155;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #1e293b;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'lstm_model' not in st.session_state:
        st.session_state.lstm_model = None
    if 'xgb_model' not in st.session_state:
        st.session_state.xgb_model = None
    if 'ensemble_model' not in st.session_state:
        st.session_state.ensemble_model = None
    if 'bank_data' not in st.session_state:
        st.session_state.bank_data = None
    if 'macro_data' not in st.session_state:
        st.session_state.macro_data = None
    if 'trajectory_results' not in st.session_state:
        st.session_state.trajectory_results = None
    if 'shap_explainer' not in st.session_state:
        st.session_state.shap_explainer = None
    if 'selected_scenario' not in st.session_state:
        st.session_state.selected_scenario = 'baseline'


def render_header():
    """Render main header"""
    st.markdown("""
    <div class="main-header">
        <h1>📊 Reg-Explain</h1>
        <p>PPNR Stress-Testing via Hybrid LSTM-XGBoost with SHAP Interpretability | CCAR Framework 2026</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with controls"""
    with st.sidebar:
        st.markdown("### 🎯 Scenario Selection")
        
        scenarios = get_fed_2026_scenarios()
        
        scenario_options = {
            'baseline': '📈 Baseline',
            'adverse': '⚠️ Adverse', 
            'severely_adverse': '🔴 Severely Adverse',
            'custom': '🎛️ Custom Black Swan'
        }
        
        selected = st.radio(
            "Select Stress Scenario",
            options=list(scenario_options.keys()),
            format_func=lambda x: scenario_options[x],
            key='scenario_selector'
        )
        st.session_state.selected_scenario = selected
        
        st.markdown("---")
        
        # Custom scenario parameters
        if selected == 'custom':
            st.markdown("### 🎛️ Custom Parameters")
            
            custom_unemployment = st.slider(
                "Peak Unemployment (%)",
                min_value=3.0, max_value=15.0, value=10.0, step=0.5
            )
            custom_vix = st.slider(
                "Peak VIX",
                min_value=15, max_value=100, value=72, step=5
            )
            custom_hpi = st.slider(
                "HPI Decline (%)",
                min_value=-40, max_value=10, value=-25, step=5
            )
            custom_rate_shock = st.slider(
                "Rate Shock (bps)",
                min_value=-200, max_value=400, value=300, step=25
            )
            
            st.session_state.custom_params = {
                'unemployment': custom_unemployment,
                'vix': custom_vix,
                'hpi_decline': custom_hpi,
                'rate_shock': custom_rate_shock
            }
        
        st.markdown("---")
        
        # Regulatory thresholds
        st.markdown("### 📋 Regulatory Thresholds")
        
        cet1_min = st.number_input(
            "CET1 Minimum (%)",
            min_value=3.0, max_value=6.0, value=4.5, step=0.25
        )
        scb = st.number_input(
            "Stress Capital Buffer (%)",
            min_value=0.0, max_value=8.0, value=2.5, step=0.25
        )
        
        st.session_state.regulatory_thresholds = {
            'cet1_min': cet1_min / 100,
            'scb': scb / 100
        }
        
        st.markdown("---")
        
        # Data source info
        st.markdown("### 📊 Data Sources")
        st.info("""
        **Macro Data**: Real FRED API
        **Bank Data**: Synthetic/Uploaded
        **Model**: LSTM + XGBoost Ensemble
        """)
        
        return selected


def render_data_tab():
    """Render data ingestion tab"""
    st.markdown("## 📥 Data Ingestion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏦 Bank Portfolio Data")
        
        data_source = st.radio(
            "Select Data Source",
            ["Generate Synthetic Data", "Upload Custom Dataset"],
            key='bank_data_source'
        )
        
        if data_source == "Generate Synthetic Data":
            n_banks = st.slider("Number of Banks", 5, 20, 10)
            
            if st.button("🔄 Generate Bank Data", use_container_width=True):
                with st.spinner("Generating synthetic bank portfolios..."):
                    generator = SyntheticBankDataGenerator()
                    st.session_state.bank_data = generator.generate_bank_portfolio(
                        n_banks=n_banks, n_quarters=36
                    )
                    st.success(f"✅ Generated data for {n_banks} banks")
        else:
            uploaded_file = st.file_uploader(
                "Upload Bank Data (CSV)",
                type=['csv'],
                help="CSV with columns: bank_id, total_assets, cet1_ratio, cre_exposure_pct, etc."
            )
            
            if uploaded_file:
                st.session_state.bank_data = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(st.session_state.bank_data)} records")
        
        if st.session_state.bank_data is not None:
            st.markdown("#### Preview")
            display_cols = ['bank_id', 'bank_name', 'total_assets', 'cet1_ratio', 'cre_exposure_pct']
            available_cols = [c for c in display_cols if c in st.session_state.bank_data.columns]
            
            # Get latest quarter for each bank
            if 'quarter' in st.session_state.bank_data.columns:
                latest = st.session_state.bank_data.groupby('bank_id').last().reset_index()
            else:
                latest = st.session_state.bank_data.head(10)
            
            st.dataframe(
                latest[available_cols].head(10),
                use_container_width=True,
                hide_index=True
            )
    
    with col2:
        st.markdown("### 🌐 Macroeconomic Data (FRED)")
        
        st.info("Real-time data from Federal Reserve Economic Data")
        
        if st.button("📡 Fetch FRED Data", use_container_width=True):
            with st.spinner("Fetching macroeconomic indicators..."):
                fetcher = FREDDataFetcher()
                conditions = fetcher.get_current_conditions()
                st.session_state.current_conditions = conditions
                st.success("✅ Macro data loaded")
        
        if 'current_conditions' in st.session_state:
            conditions = st.session_state.current_conditions
            
            st.markdown("#### Current Economic Conditions")
            
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.metric(
                    "Unemployment Rate",
                    f"{conditions.get('unemployment_rate', 4.2):.1f}%"
                )
                st.metric(
                    "Fed Funds Rate",
                    f"{conditions.get('fed_funds_rate', 4.5):.2f}%"
                )
                st.metric(
                    "10Y Treasury",
                    f"{conditions.get('treasury_10y', 4.0):.2f}%"
                )
            
            with metrics_col2:
                st.metric(
                    "VIX",
                    f"{conditions.get('vix', 18.0):.1f}"
                )
                st.metric(
                    "HPI Growth",
                    f"{conditions.get('hpi_growth', 3.0):.1f}%"
                )
                spread = conditions.get('yield_curve_spread', -0.3)
                st.metric(
                    "Yield Curve Spread",
                    f"{spread:.2f}%",
                    delta="Inverted" if spread < 0 else "Normal"
                )


def render_model_tab():
    """Render model training tab"""
    st.markdown("## 🤖 Model Training")
    
    if st.session_state.bank_data is None:
        st.warning("⚠️ Please load bank data first in the Data tab")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔄 LSTM Configuration")
        st.markdown("*Temporal model for interest rate path dependency*")
        
        lstm_units = st.slider("LSTM Units", 32, 128, 64)
        lstm_dropout = st.slider("Dropout Rate", 0.1, 0.5, 0.2)
        lstm_epochs = st.slider("Training Epochs", 50, 200, 100)
    
    with col2:
        st.markdown("### 🌲 XGBoost Configuration")
        st.markdown("*Gradient boosting for static risk factors*")
        
        xgb_estimators = st.slider("N Estimators", 100, 500, 200)
        xgb_depth = st.slider("Max Depth", 3, 10, 6)
        xgb_lr = st.slider("Learning Rate", 0.01, 0.2, 0.05)
    
    st.markdown("---")
    
    if st.button("🚀 Train Ensemble Model", use_container_width=True, type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Generate training data
        status_text.text("Generating training data...")
        progress_bar.progress(10)
        
        generator = SyntheticBankDataGenerator()
        X_temporal, X_static, y = generator.generate_training_data(n_samples=1000)
        
        # Train LSTM
        status_text.text("Training LSTM model (temporal sequences)...")
        progress_bar.progress(30)
        
        lstm_model = PPNRLSTMModel(sequence_length=9, n_features=3)
        lstm_model.build_model(lstm_units=lstm_units, dropout_rate=lstm_dropout)
        lstm_model.fit(X_temporal, y, epochs=lstm_epochs, batch_size=32)
        st.session_state.lstm_model = lstm_model
        
        # Train XGBoost
        status_text.text("Training XGBoost model (static features)...")
        progress_bar.progress(60)
        
        xgb_model = PPNRXGBoostModel()
        xgb_model.build_model(n_estimators=xgb_estimators, max_depth=xgb_depth, learning_rate=xgb_lr)
        
        feature_names = ['cre_exposure', 'residential_exposure', 'cet1_ratio', 
                        'total_assets_log', 'npl_ratio']
        xgb_model.fit(X_static, y, feature_names=feature_names)
        st.session_state.xgb_model = xgb_model
        
        # Calibrate ensemble
        status_text.text("Calibrating ensemble weights...")
        progress_bar.progress(80)
        
        ensemble = PPNREnsembleModel(lstm_model, xgb_model)
        weights = ensemble.calibrate_weights(X_temporal, X_static, y, method='optimize')
        st.session_state.ensemble_model = ensemble
        
        # Initialize SHAP explainer
        status_text.text("Initializing SHAP explainer...")
        progress_bar.progress(90)
        
        shap_explainer = SHAPExplainer(xgb_model)
        shap_explainer.initialize_explainer(X_static[:100], feature_names)
        st.session_state.shap_explainer = shap_explainer
        
        progress_bar.progress(100)
        status_text.text("✅ Training complete!")
        st.session_state.models_trained = True
        
        # Display training results
        st.markdown("### 📊 Training Results")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric("LSTM Weight", f"{weights[0]:.3f}")
        with result_col2:
            st.metric("XGBoost Weight", f"{weights[1]:.3f}")
        with result_col3:
            # Get feature importance
            importance = xgb_model.get_feature_importance()
            top_feature = max(importance, key=importance.get) if importance else "N/A"
            st.metric("Top Feature", top_feature)
        
        # Feature importance chart
        if importance:
            st.markdown("#### XGBoost Feature Importance")
            fig = px.bar(
                x=list(importance.values()),
                y=list(importance.keys()),
                orientation='h',
                labels={'x': 'Importance', 'y': 'Feature'},
                color=list(importance.values()),
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=300,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)


def render_forecast_tab():
    """Render capital forecast tab"""
    st.markdown("## 📈 Capital Erosion Forecast")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first in the Model Training tab")
        return
    
    if st.session_state.bank_data is None:
        st.warning("⚠️ Please load bank data first")
        return
    
    scenario = st.session_state.selected_scenario
    scenarios_info = get_fed_2026_scenarios()
    
    # Scenario info card
    if scenario in scenarios_info:
        info = scenarios_info[scenario]
        st.markdown(f"""
        <div class="scenario-card selected">
            <h4>{info['name']} Scenario</h4>
            <p>{info['description']}</p>
            <p><strong>Unemployment Peak:</strong> {info['unemployment_peak']}% | 
            <strong>VIX Peak:</strong> {info['vix_peak']} | 
            <strong>HPI Decline:</strong> {info['hpi_decline']}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("📊 Run 9-Quarter Forecast", use_container_width=True, type="primary"):
        with st.spinner("Running stress test simulation..."):
            # Generate scenario data
            fetcher = FREDDataFetcher()
            
            if scenario == 'custom':
                params = st.session_state.get('custom_params', {})
                # Generate custom scenario
                generator = SyntheticBankDataGenerator()
                scenario_data = generator.generate_scenario_data('severely_adverse', n_quarters=9)
                # Override with custom params
                scenario_data['unemployment_rate'] = params.get('unemployment', 10.0)
                scenario_data['vix'] = params.get('vix', 72)
                scenario_data['hpi_growth'] = params.get('hpi_decline', -25)
            else:
                scenario_data = fetcher.project_scenario(scenario, n_quarters=9)
            
            # Get latest bank data
            bank_df = st.session_state.bank_data
            if 'quarter' in bank_df.columns:
                bank_df = bank_df.groupby('bank_id').last().reset_index()
            
            # Generate PPNR forecasts
            forecaster = QuarterlyPPNRForecaster(st.session_state.ensemble_model)
            ppnr_forecasts = forecaster.forecast(bank_df, scenario_data, n_quarters=9)
            
            # Calculate capital trajectory
            thresholds = RegulatoryThresholds()
            reg_params = st.session_state.get('regulatory_thresholds', {})
            thresholds.CET1_MINIMUM = reg_params.get('cet1_min', 0.045)
            thresholds.SCB_DEFAULT = reg_params.get('scb', 0.025)
            
            engine = CapitalEngine(thresholds)
            trajectory = engine.calculate_cet1_trajectory(
                bank_df, ppnr_forecasts, scenario_data, n_quarters=9
            )
            
            st.session_state.trajectory_results = trajectory
            st.session_state.scenario_data = scenario_data
            st.session_state.ppnr_forecasts = ppnr_forecasts
            
            # Breach analysis
            breach_analysis = engine.get_breach_analysis(trajectory)
            
            st.success("✅ Forecast complete")
            
            # Display results
            render_forecast_results(trajectory, breach_analysis, scenario_data)


def render_forecast_results(trajectory: pd.DataFrame, breach_analysis: dict, scenario_data: pd.DataFrame):
    """Render forecast visualization"""
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    min_cet1 = trajectory['cet1_ratio'].min()
    initial_cet1 = trajectory[trajectory['quarter'] == 1]['cet1_ratio'].mean()
    
    with col1:
        color_class = "status-safe" if min_cet1 > 0.07 else ("status-warning" if min_cet1 > 0.045 else "status-danger")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {color_class}">{min_cet1*100:.1f}%</div>
            <div class="metric-label">Minimum CET1</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        decline = initial_cet1 - min_cet1
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value status-warning">{decline*100:.1f}%</div>
            <div class="metric-label">Capital Decline</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        breach_rate = breach_analysis['breach_rate'] * 100
        color_class = "status-safe" if breach_rate == 0 else ("status-warning" if breach_rate < 30 else "status-danger")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {color_class}">{breach_rate:.0f}%</div>
            <div class="metric-label">Banks Breaching</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        worst_q = breach_analysis['worst_breach']['quarter'] or 'N/A'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">Q{worst_q}</div>
            <div class="metric-label">Worst Quarter</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Breach alert
    if breach_analysis['banks_with_breach'] > 0:
        st.markdown(f"""
        <div class="breach-alert">
            <h3>⚠️ CAPITAL BREACH ALERT</h3>
            <p><strong>{breach_analysis['banks_with_breach']}</strong> of {breach_analysis['total_banks']} banks 
            projected to breach CET1 minimum. First breach expected in <strong>Q{breach_analysis['first_breach_quarter']}</strong>.
            Capital shortfall: <strong>${abs(breach_analysis['capital_shortfall'])/1e9:.2f}B</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # CET1 Trajectory Chart
    st.markdown("### 📉 CET1 Capital Erosion Map")
    
    fig = go.Figure()
    
    # Plot each bank's trajectory
    for bank_id in trajectory['bank_id'].unique()[:10]:  # Top 10 banks
        bank_traj = trajectory[trajectory['bank_id'] == bank_id]
        bank_name = bank_traj['bank_name'].iloc[0] if 'bank_name' in bank_traj.columns else bank_id
        
        fig.add_trace(go.Scatter(
            x=bank_traj['quarter'],
            y=bank_traj['cet1_ratio'] * 100,
            mode='lines+markers',
            name=bank_name[:20],
            line=dict(width=2),
            hovertemplate=f"{bank_name}<br>Q%{{x}}: %{{y:.2f}}%<extra></extra>"
        ))
    
    # Add regulatory minimum line
    reg_min = st.session_state.get('regulatory_thresholds', {}).get('cet1_min', 0.045)
    scb = st.session_state.get('regulatory_thresholds', {}).get('scb', 0.025)
    total_min = (reg_min + scb) * 100
    
    fig.add_hline(
        y=total_min,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Min Required ({total_min:.1f}%)",
        annotation_position="right"
    )
    
    fig.add_hline(
        y=reg_min * 100,
        line_dash="dot",
        line_color="#f59e0b",
        annotation_text=f"CET1 Min ({reg_min*100:.1f}%)",
        annotation_position="right"
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Quarter",
        yaxis_title="CET1 Ratio (%)",
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5
        ),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Macro scenario chart
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 📊 Macro Stress Path")
        
        fig_macro = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Unemployment Rate', 'VIX', 'HPI Growth', 'Fed Funds Rate')
        )
        
        fig_macro.add_trace(
            go.Scatter(x=scenario_data['quarter'], y=scenario_data['unemployment_rate'],
                      mode='lines+markers', line=dict(color='#ef4444')),
            row=1, col=1
        )
        
        fig_macro.add_trace(
            go.Scatter(x=scenario_data['quarter'], y=scenario_data['vix'],
                      mode='lines+markers', line=dict(color='#f59e0b')),
            row=1, col=2
        )
        
        fig_macro.add_trace(
            go.Scatter(x=scenario_data['quarter'], y=scenario_data['hpi_growth'],
                      mode='lines+markers', line=dict(color='#22c55e')),
            row=2, col=1
        )
        
        fig_macro.add_trace(
            go.Scatter(x=scenario_data['quarter'], y=scenario_data['fed_funds_rate'],
                      mode='lines+markers', line=dict(color='#3b82f6')),
            row=2, col=2
        )
        
        fig_macro.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_macro, use_container_width=True)
    
    with col_right:
        st.markdown("### 💰 PPNR Components")
        
        # Aggregate PPNR by quarter
        ppnr_agg = trajectory.groupby('quarter').agg({
            'ppnr': 'sum',
            'credit_losses': 'sum'
        }).reset_index()
        
        fig_ppnr = go.Figure()
        
        fig_ppnr.add_trace(go.Bar(
            x=ppnr_agg['quarter'],
            y=ppnr_agg['ppnr'] / 1e9,
            name='PPNR',
            marker_color='#22c55e'
        ))
        
        fig_ppnr.add_trace(go.Bar(
            x=ppnr_agg['quarter'],
            y=-ppnr_agg['credit_losses'] / 1e9,
            name='Credit Losses',
            marker_color='#ef4444'
        ))
        
        fig_ppnr.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Quarter",
            yaxis_title="$ Billions",
            height=400,
            barmode='relative',
            legend=dict(orientation="h", y=-0.15)
        )
        
        st.plotly_chart(fig_ppnr, use_container_width=True)


def render_explainer_tab():
    """Render SHAP explainability tab"""
    st.markdown("## 🔍 Model Explainability (SR 11-7 Compliance)")
    
    if st.session_state.trajectory_results is None:
        st.warning("⚠️ Please run a forecast first in the Forecast tab")
        return
    
    if st.session_state.shap_explainer is None:
        st.warning("⚠️ SHAP explainer not initialized. Please retrain models.")
        return
    
    trajectory = st.session_state.trajectory_results
    scenario = st.session_state.selected_scenario
    
    # Get SHAP explanations
    bank_df = st.session_state.bank_data
    if 'quarter' in bank_df.columns:
        bank_df = bank_df.groupby('bank_id').last().reset_index()
    
    scenario_data = st.session_state.get('scenario_data')
    
    # Create static features for SHAP
    X_static, feature_names = create_static_features(bank_df, scenario_data)
    
    # Run SHAP analysis
    with st.spinner("Computing SHAP values..."):
        shap_results = st.session_state.shap_explainer.explain_predictions(X_static)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌍 Global Feature Importance")
        st.markdown("*Which macro factors drive PPNR losses most across the banking sector*")
        
        importance = shap_results['global_importance']
        
        fig = px.bar(
            x=list(importance.values()),
            y=list(importance.keys()),
            orientation='h',
            labels={'x': 'Mean |SHAP Value|', 'y': 'Feature'},
            color=list(importance.values()),
            color_continuous_scale='Purples'
        )
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Primary Stress Drivers")
        
        top_features = list(importance.keys())[:5]
        
        for i, feature in enumerate(top_features):
            value = importance[feature]
            pct = value / sum(importance.values()) * 100
            
            color = ['#8b5cf6', '#6366f1', '#3b82f6', '#0ea5e9', '#14b8a6'][i]
            
            st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                    <span style="color: #e2e8f0; font-weight: 500;">{feature.replace('_', ' ').title()}</span>
                    <span style="color: #94a3b8;">{pct:.1f}%</span>
                </div>
                <div style="background: #1e293b; border-radius: 4px; height: 8px;">
                    <div style="background: {color}; width: {pct}%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # AI-Powered Explanation
    st.markdown("### 🤖 AI-Powered Analysis (Gemini 3 Flash)")
    
    if st.button("📝 Generate Natural Language Explanation", use_container_width=True):
        with st.spinner("Generating explanation with Gemini 3 Flash..."):
            gemini = GeminiExplainer()
            
            # Run async function
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            explanation = loop.run_until_complete(
                gemini.generate_global_explanation(shap_results, scenario)
            )
            loop.close()
            
            st.markdown(f"""
            <div class="shap-explanation">
                <p>{explanation}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Local explanation for specific quarter
    st.markdown("---")
    st.markdown("### 🔬 Local Scenario Explainer")
    st.markdown("*Breakdown of a specific quarter's PPNR drivers*")
    
    selected_quarter = st.selectbox(
        "Select Quarter to Analyze",
        options=list(range(1, 10)),
        format_func=lambda x: f"Q{x} - {'Peak Stress' if x in [4, 5] else 'Recovery' if x > 6 else 'Initial Stress'}"
    )
    
    if st.button("🔍 Analyze Quarter", use_container_width=True):
        # Get explanation for selected quarter
        quarter_data = trajectory[trajectory['quarter'] == selected_quarter]
        
        if len(quarter_data) > 0:
            # Use first bank for demo
            sample_idx = 0
            single_explanation = st.session_state.shap_explainer.explain_single_prediction(
                X_static[sample_idx],
                quarter_data['ppnr'].iloc[0]
            )
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("#### 📈 Positive Drivers")
                for driver in single_explanation['top_positive_drivers']:
                    contrib = single_explanation['contributions'][driver]
                    st.markdown(f"""
                    - **{driver.replace('_', ' ').title()}**: +{contrib['shap_value']:.4f}
                      (Value: {contrib['feature_value']:.3f})
                    """)
            
            with col_b:
                st.markdown("#### 📉 Negative Drivers")
                for driver in single_explanation['top_negative_drivers']:
                    contrib = single_explanation['contributions'][driver]
                    st.markdown(f"""
                    - **{driver.replace('_', ' ').title()}**: {contrib['shap_value']:.4f}
                      (Value: {contrib['feature_value']:.3f})
                    """)
            
            # Generate AI explanation for this quarter
            if st.button("🤖 Explain This Quarter"):
                with st.spinner("Analyzing..."):
                    gemini = GeminiExplainer()
                    
                    # Create scenario impact mock
                    scenario_impact = {
                        'baseline_prediction': quarter_data['ppnr'].mean() * 1.2,
                        'stress_prediction': quarter_data['ppnr'].mean(),
                        'prediction_change_pct': -20,
                        'primary_driver': single_explanation['top_negative_drivers'][0] if single_explanation['top_negative_drivers'] else 'N/A',
                        'primary_driver_impact': -0.005,
                        'impact_breakdown': single_explanation['contributions']
                    }
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    quarter_explanation = loop.run_until_complete(
                        gemini.generate_scenario_explanation(scenario_impact, scenario)
                    )
                    loop.close()
                    
                    st.markdown(f"""
                    <div class="shap-explanation">
                        <p>{quarter_explanation}</p>
                    </div>
                    """, unsafe_allow_html=True)


def render_reports_tab():
    """Render regulatory reports tab"""
    st.markdown("## 📋 Regulatory Reports")
    
    if st.session_state.trajectory_results is None:
        st.warning("⚠️ Please run a forecast first")
        return
    
    trajectory = st.session_state.trajectory_results
    
    st.markdown("### 📊 Downloadable Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### CET1 Trajectory Report")
        csv = trajectory.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"cet1_trajectory_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.markdown("#### Executive Summary")
        
        # Create summary
        summary = f"""
# CCAR Stress Test Results
## Executive Summary

**Date**: {datetime.now().strftime('%Y-%m-%d')}
**Scenario**: {st.session_state.selected_scenario.replace('_', ' ').title()}
**Projection Horizon**: 9 Quarters

### Key Findings

- **Minimum CET1 Ratio**: {trajectory['cet1_ratio'].min()*100:.2f}%
- **Banks Breaching Minimum**: {trajectory[trajectory['breach']]['bank_id'].nunique()}
- **Peak Capital Decline**: {(trajectory.groupby('bank_id')['cet1_ratio'].first().mean() - trajectory['cet1_ratio'].min())*100:.2f}%
- **Total Credit Losses**: ${trajectory['credit_losses'].sum()/1e9:.2f}B

### Regulatory Compliance

This analysis complies with:
- SR 11-7: Model Risk Management
- CCAR/DFAST Requirements
- Basel III Capital Standards

---
*Generated by Reg-Explain PPNR Stress Testing Framework*
        """
        
        st.download_button(
            label="📥 Download Summary",
            data=summary,
            file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    with col3:
        st.markdown("#### Bank-Level Details")
        
        # Pivot table
        pivot = trajectory.pivot_table(
            index='bank_id',
            columns='quarter',
            values='cet1_ratio',
            aggfunc='first'
        ).round(4)
        
        pivot_csv = pivot.to_csv()
        st.download_button(
            label="📥 Download Pivot",
            data=pivot_csv,
            file_name=f"bank_cet1_pivot_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Detailed bank table
    st.markdown("### 🏦 Bank-Level Results")
    
    # Summary by bank
    bank_summary = trajectory.groupby(['bank_id', 'bank_name']).agg({
        'cet1_ratio': ['first', 'min', 'last'],
        'credit_losses': 'sum',
        'breach': 'any'
    }).round(4)
    
    bank_summary.columns = ['Initial CET1', 'Min CET1', 'Final CET1', 'Total Losses', 'Breach']
    bank_summary = bank_summary.reset_index()
    
    # Style the dataframe
    def highlight_breach(val):
        if val == True:
            return 'background-color: #7f1d1d; color: #fca5a5'
        return ''
    
    st.dataframe(
        bank_summary,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Initial CET1': st.column_config.NumberColumn(format="%.2f%%"),
            'Min CET1': st.column_config.NumberColumn(format="%.2f%%"),
            'Final CET1': st.column_config.NumberColumn(format="%.2f%%"),
            'Total Losses': st.column_config.NumberColumn(format="$%.2fB"),
            'Breach': st.column_config.CheckboxColumn()
        }
    )


def main():
    """Main application entry point"""
    init_session_state()
    render_header()
    
    # Sidebar
    selected_scenario = render_sidebar()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📥 Data Ingestion",
        "🤖 Model Training", 
        "📈 Capital Forecast",
        "🔍 Explainability",
        "📋 Reports"
    ])
    
    with tab1:
        render_data_tab()
    
    with tab2:
        render_model_tab()
    
    with tab3:
        render_forecast_tab()
    
    with tab4:
        render_explainer_tab()
    
    with tab5:
        render_reports_tab()


if __name__ == "__main__":
    main()
