"""
Synthetic Bank Data Generator for PPNR Stress Testing
Generates realistic bank portfolio data with regulatory metrics
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class SyntheticBankDataGenerator:
    """Generates synthetic bank data for stress testing"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
    def generate_bank_portfolio(self, n_banks=10, n_quarters=36):
        """Generate synthetic bank portfolio data"""
        banks = []
        
        bank_names = [
            "First National Bank", "Metro Commercial Bank", "Pacific Trust Bank",
            "Atlantic Financial Corp", "Midwest Regional Bank", "Coastal Savings Bank",
            "Mountain State Bank", "Valley Commerce Bank", "Harbor Financial Group",
            "Prairie National Bank", "Desert Southwest Bank", "Northern Trust Corp"
        ]
        
        for i in range(n_banks):
            bank_name = bank_names[i % len(bank_names)]
            
            # Base characteristics
            total_assets = np.random.uniform(50, 500) * 1e9  # $50B - $500B
            cre_exposure_pct = np.random.uniform(0.05, 0.35)  # 5-35% CRE exposure
            residential_exposure_pct = np.random.uniform(0.15, 0.45)
            
            for q in range(n_quarters):
                quarter_date = datetime(2020, 1, 1) + timedelta(days=q*91)
                
                # CET1 Capital with some variation
                cet1_ratio = np.random.normal(0.12, 0.02)  # ~12% average
                cet1_ratio = max(0.06, min(0.18, cet1_ratio))
                
                # Risk-Weighted Assets
                rwa = total_assets * np.random.uniform(0.6, 0.85)
                cet1_capital = cet1_ratio * rwa
                
                # PPNR components
                net_interest_income = total_assets * np.random.uniform(0.025, 0.04) / 4
                non_interest_income = total_assets * np.random.uniform(0.005, 0.015) / 4
                non_interest_expense = total_assets * np.random.uniform(0.015, 0.025) / 4
                ppnr = net_interest_income + non_interest_income - non_interest_expense
                
                # Loan portfolio breakdown
                cre_loans = total_assets * cre_exposure_pct
                residential_loans = total_assets * residential_exposure_pct
                consumer_loans = total_assets * np.random.uniform(0.05, 0.15)
                commercial_loans = total_assets * np.random.uniform(0.1, 0.25)
                
                banks.append({
                    'bank_id': f'BANK_{i:03d}',
                    'bank_name': bank_name,
                    'quarter': quarter_date.strftime('%Y-Q%q').replace('Q%q', f'Q{(quarter_date.month-1)//3+1}'),
                    'quarter_date': quarter_date,
                    'total_assets': total_assets,
                    'rwa': rwa,
                    'cet1_capital': cet1_capital,
                    'cet1_ratio': cet1_ratio,
                    'ppnr': ppnr,
                    'net_interest_income': net_interest_income,
                    'non_interest_income': non_interest_income,
                    'non_interest_expense': non_interest_expense,
                    'cre_exposure': cre_loans,
                    'cre_exposure_pct': cre_exposure_pct,
                    'residential_exposure': residential_loans,
                    'residential_exposure_pct': residential_exposure_pct,
                    'consumer_loans': consumer_loans,
                    'commercial_loans': commercial_loans,
                    'loan_loss_provision': ppnr * np.random.uniform(0.05, 0.15),
                    'credit_card_utilization': np.random.uniform(0.15, 0.35),
                    'npl_ratio': np.random.uniform(0.005, 0.025),
                    'tier1_leverage_ratio': cet1_ratio * np.random.uniform(0.7, 0.9)
                })
        
        return pd.DataFrame(banks)
    
    def generate_scenario_data(self, scenario_type='baseline', n_quarters=9):
        """Generate macro scenario data for 9-quarter projection"""
        
        scenarios = {
            'baseline': {
                'unemployment_rate': (4.5, 0.3),
                'gdp_growth': (2.0, 0.5),
                'fed_funds_rate': (4.5, 0.25),
                'treasury_10y': (4.0, 0.2),
                'treasury_3m': (4.3, 0.15),
                'vix': (18, 3),
                'hpi_growth': (3.0, 1.0),
                'credit_card_util': (0.22, 0.02),
                'real_dpi_growth': (1.5, 0.5)
            },
            'adverse': {
                'unemployment_rate': (7.0, 1.0),
                'gdp_growth': (-1.0, 0.8),
                'fed_funds_rate': (2.0, 0.5),
                'treasury_10y': (2.5, 0.3),
                'treasury_3m': (1.5, 0.25),
                'vix': (45, 8),
                'hpi_growth': (-5.0, 2.0),
                'credit_card_util': (0.30, 0.03),
                'real_dpi_growth': (-2.0, 1.0)
            },
            'severely_adverse': {
                'unemployment_rate': (10.0, 1.5),
                'gdp_growth': (-4.0, 1.0),
                'fed_funds_rate': (0.5, 0.25),
                'treasury_10y': (1.5, 0.3),
                'treasury_3m': (0.25, 0.1),
                'vix': (72, 10),
                'hpi_growth': (-15.0, 3.0),
                'credit_card_util': (0.38, 0.04),
                'real_dpi_growth': (-5.0, 1.5)
            }
        }
        
        params = scenarios.get(scenario_type, scenarios['baseline'])
        
        scenario_data = []
        base_date = datetime(2026, 1, 1)
        
        for q in range(n_quarters):
            quarter_date = base_date + timedelta(days=q*91)
            
            # Add path dependency (cumulative stress)
            stress_multiplier = 1 + (q / n_quarters) * 0.3 if scenario_type != 'baseline' else 1
            
            row = {
                'quarter': q + 1,
                'quarter_date': quarter_date,
                'scenario': scenario_type,
                'unemployment_rate': np.clip(
                    np.random.normal(params['unemployment_rate'][0] * stress_multiplier, 
                                   params['unemployment_rate'][1]), 
                    2.5, 15
                ),
                'gdp_growth': np.random.normal(
                    params['gdp_growth'][0] / stress_multiplier if scenario_type != 'baseline' else params['gdp_growth'][0],
                    params['gdp_growth'][1]
                ),
                'fed_funds_rate': np.clip(
                    np.random.normal(params['fed_funds_rate'][0], params['fed_funds_rate'][1]),
                    0, 8
                ),
                'treasury_10y': np.clip(
                    np.random.normal(params['treasury_10y'][0], params['treasury_10y'][1]),
                    0.5, 8
                ),
                'treasury_3m': np.clip(
                    np.random.normal(params['treasury_3m'][0], params['treasury_3m'][1]),
                    0, 7
                ),
                'vix': np.clip(
                    np.random.normal(params['vix'][0] * stress_multiplier, params['vix'][1]),
                    10, 100
                ),
                'hpi_growth': np.random.normal(
                    params['hpi_growth'][0] / stress_multiplier if scenario_type != 'baseline' else params['hpi_growth'][0],
                    params['hpi_growth'][1]
                ),
                'credit_card_utilization': np.clip(
                    np.random.normal(params['credit_card_util'][0] * stress_multiplier, params['credit_card_util'][1]),
                    0.1, 0.6
                ),
                'real_dpi_growth': np.random.normal(
                    params['real_dpi_growth'][0] / stress_multiplier if scenario_type != 'baseline' else params['real_dpi_growth'][0],
                    params['real_dpi_growth'][1]
                )
            }
            
            # Calculate yield curve spread (10Y - 3M)
            row['yield_curve_spread'] = row['treasury_10y'] - row['treasury_3m']
            
            scenario_data.append(row)
        
        return pd.DataFrame(scenario_data)
    
    def generate_training_data(self, n_samples=1000, sequence_length=9):
        """Generate training data for ML models"""
        
        X_temporal = []  # For LSTM
        X_static = []    # For XGBoost
        y = []           # PPNR outcomes
        
        for _ in range(n_samples):
            # Temporal features (9 quarters of rates)
            base_rate = np.random.uniform(0.5, 6)
            rate_path = [base_rate]
            for _ in range(sequence_length - 1):
                change = np.random.normal(0, 0.3)
                rate_path.append(np.clip(rate_path[-1] + change, 0, 8))
            
            # Add more temporal features
            vix_path = np.random.uniform(12, 70, sequence_length)
            spread_path = np.random.uniform(-1, 3, sequence_length)
            
            temporal = np.column_stack([rate_path, vix_path, spread_path])
            X_temporal.append(temporal)
            
            # Static features
            cre_exposure = np.random.uniform(0.05, 0.35)
            residential_exposure = np.random.uniform(0.15, 0.45)
            cet1_ratio = np.random.uniform(0.08, 0.16)
            total_assets_log = np.log(np.random.uniform(50, 500) * 1e9)
            npl_ratio = np.random.uniform(0.005, 0.03)
            
            static = [cre_exposure, residential_exposure, cet1_ratio, 
                     total_assets_log, npl_ratio]
            X_static.append(static)
            
            # Target: PPNR change (influenced by features)
            rate_impact = -0.1 * (rate_path[-1] - rate_path[0])
            vix_impact = -0.002 * np.mean(vix_path)
            cre_impact = -0.15 * cre_exposure * (vix_path[-1] > 40)
            base_ppnr = 0.008 * np.exp(total_assets_log - 25)
            
            ppnr_change = base_ppnr * (1 + rate_impact + vix_impact + cre_impact)
            ppnr_change += np.random.normal(0, 0.001)
            
            y.append(ppnr_change)
        
        return np.array(X_temporal), np.array(X_static), np.array(y)


def get_fed_2026_scenarios():
    """Return the Fed 2026 macro scenarios as specified"""
    return {
        'baseline': {
            'name': 'Baseline',
            'unemployment_peak': 4.5,
            'vix_peak': 20,
            'gdp_trough': 2.0,
            'hpi_decline': 0,
            'description': 'Normal economic conditions with moderate growth'
        },
        'adverse': {
            'name': 'Adverse',
            'unemployment_peak': 7.5,
            'vix_peak': 50,
            'gdp_trough': -2.0,
            'hpi_decline': -10,
            'description': 'Moderate recession with elevated stress'
        },
        'severely_adverse': {
            'name': 'Severely Adverse',
            'unemployment_peak': 10.0,
            'vix_peak': 72,
            'gdp_trough': -6.0,
            'hpi_decline': -25,
            'description': 'Severe recession with major asset price declines'
        }
    }
