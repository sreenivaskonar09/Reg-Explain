# Reg-Explain: PPNR Stress-Testing Framework

## Original Problem Statement
Build a next-generation CCAR (Comprehensive Capital Analysis and Review) framework using a hybrid LSTM-XGBoost ensemble model to forecast Pre-Provision Net Revenue (PPNR) over a 9-quarter regulatory horizon with full SHAP interpretability for SR 11-7 compliance.

## User Personas
1. **Bank Risk Officers**: Monitor capital adequacy and stress test results
2. **Federal Reserve Examiners**: Review model outputs for regulatory compliance
3. **Model Risk Management Teams**: Validate model interpretability (SR 11-7)
4. **C-Suite Executives**: Strategic capital planning decisions

## Core Requirements
- Hybrid LSTM-XGBoost ensemble for PPNR forecasting
- 9-quarter capital erosion projection (CET1 ratio)
- Fed 2026 stress scenarios (Baseline, Adverse, Severely Adverse)
- SHAP-based model explainability
- AI-powered natural language explanations (Gemini 3 Flash)
- Real FRED macro data integration
- Synthetic bank portfolio generation

## What's Been Implemented (December 2025)

### Backend (FastAPI)
- ✅ Data ingestion endpoints (synthetic banks, FRED macro data)
- ✅ LSTM model for temporal sequences (interest rate paths)
- ✅ XGBoost model for static risk factors (CRE exposure, CET1)
- ✅ Ensemble calibration with optimized weights
- ✅ Capital Engine (CET1 trajectory, breach detection)
- ✅ SHAP explainability module
- ✅ Gemini 3 Flash AI explanations

### Frontend (React)
- ✅ Data Ingestion tab (bank data, FRED macro)
- ✅ Model Training tab (LSTM/XGBoost configuration)
- ✅ Capital Forecast tab (charts, metrics, alerts)
- ✅ Explainability tab (SHAP, AI analysis)
- ✅ Reports tab (CSV downloads)
- ✅ Scenario selector (Baseline, Adverse, Severely Adverse)
- ✅ Bloomberg Terminal dark theme (verified December 2025)

### Key Features
- Real-time macro indicators from FRED API
- Interactive model training with configurable hyperparameters
- Visual capital erosion maps with regulatory threshold lines
- Breach alerts with SR 11-7 compliance messaging

## Prioritized Backlog

### P0 (Critical)
- All critical features implemented ✅

### P1 (High Priority)
- [ ] Custom Black Swan scenario builder
- [ ] Multi-bank comparison views
- [ ] Historical stress test result database

### P2 (Medium Priority)
- [ ] PDF report generation
- [ ] Email alerts for breach conditions
- [ ] User authentication and role-based access

### P3 (Future)
- [ ] API rate limiting
- [ ] Model versioning and A/B testing
- [ ] Integration with bank core systems

## Architecture
```
Frontend (React) → FastAPI Backend → MongoDB
                         ↓
    [LSTM Model] + [XGBoost Model] → Ensemble
                         ↓
         Capital Engine → SHAP Explainer → Gemini AI
```

## Next Tasks
1. Implement custom scenario parameter persistence
2. Add bank data CSV upload with validation
3. Create executive summary PDF export
4. Add model comparison dashboard
