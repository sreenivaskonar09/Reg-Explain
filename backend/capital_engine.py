"""
Capital Engine: CET1 Ratio Calculations and Regulatory Compliance
Implements the capital erosion calculations for CCAR stress testing
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RegulatoryThresholds:
    """Federal Reserve regulatory capital minimums"""
    CET1_MINIMUM: float = 0.045  # 4.5%
    TIER1_MINIMUM: float = 0.06  # 6%
    TOTAL_CAPITAL_MINIMUM: float = 0.08  # 8%
    LEVERAGE_RATIO_MINIMUM: float = 0.04  # 4%
    
    # Stress Capital Buffer (bank-specific, typically 2.5-8%)
    SCB_DEFAULT: float = 0.025  # 2.5% default
    
    # G-SIB Surcharge (for systemically important banks)
    GSIB_SURCHARGE: float = 0.0  # 0-3.5% depending on bank size


class CapitalEngine:
    """
    Capital calculation engine for CCAR stress testing
    Calculates CET1 erosion over 9-quarter horizon
    """
    
    def __init__(self, thresholds: Optional[RegulatoryThresholds] = None):
        self.thresholds = thresholds or RegulatoryThresholds()
        
    def calculate_cet1_trajectory(self, 
                                  bank_data: pd.DataFrame,
                                  ppnr_forecasts: pd.DataFrame,
                                  scenario_data: pd.DataFrame,
                                  n_quarters: int = 9) -> pd.DataFrame:
        """
        Calculate CET1 ratio trajectory over stress horizon
        
        CET1 Ratio = (CET1 Capital + PPNR - Losses - Dividends) / RWA
        """
        trajectories = []
        
        for _, bank in bank_data.iterrows():
            bank_id = bank.get('bank_id', 'BANK_000')
            bank_name = bank.get('bank_name', 'Unknown')
            
            # Initial values
            cet1_capital = bank.get('cet1_capital', bank.get('total_assets', 100e9) * 0.12)
            rwa = bank.get('rwa', bank.get('total_assets', 100e9) * 0.7)
            cre_exposure = bank.get('cre_exposure', bank.get('total_assets', 100e9) * 0.2)
            residential_exposure = bank.get('residential_exposure', bank.get('total_assets', 100e9) * 0.3)
            
            # SCB for this bank (can be customized)
            scb = bank.get('stress_capital_buffer', self.thresholds.SCB_DEFAULT)
            
            for q in range(1, n_quarters + 1):
                # Get PPNR for this quarter
                ppnr_row = ppnr_forecasts[
                    (ppnr_forecasts['bank_id'] == bank_id) & 
                    (ppnr_forecasts['quarter'] == q)
                ]
                
                if len(ppnr_row) > 0:
                    quarterly_ppnr = ppnr_row['ppnr_forecast'].values[0]
                else:
                    quarterly_ppnr = cet1_capital * 0.02  # Default 2% of capital
                
                # Get macro stress for this quarter
                macro_q = scenario_data[scenario_data['quarter'] == q]
                if len(macro_q) > 0:
                    macro = macro_q.iloc[0]
                else:
                    macro = {}
                
                # Calculate credit losses based on scenario
                credit_losses = self._calculate_credit_losses(
                    cre_exposure, 
                    residential_exposure,
                    bank.get('consumer_loans', 0),
                    macro
                )
                
                # RWA adjustments (increases in stress)
                rwa_growth_factor = self._calculate_rwa_growth(macro, q)
                rwa = rwa * rwa_growth_factor
                
                # Dividends and buybacks (reduced in stress)
                dividends = self._calculate_dividends(cet1_capital, macro)
                
                # Update CET1 Capital
                cet1_capital = cet1_capital + quarterly_ppnr - credit_losses - dividends
                
                # Calculate ratios
                cet1_ratio = cet1_capital / rwa if rwa > 0 else 0
                
                # Determine breach status
                min_required = self.thresholds.CET1_MINIMUM + scb
                breach = cet1_ratio < min_required
                buffer_remaining = cet1_ratio - min_required
                
                trajectories.append({
                    'bank_id': bank_id,
                    'bank_name': bank_name,
                    'quarter': q,
                    'cet1_capital': cet1_capital,
                    'rwa': rwa,
                    'cet1_ratio': cet1_ratio,
                    'ppnr': quarterly_ppnr,
                    'credit_losses': credit_losses,
                    'dividends': dividends,
                    'scb': scb,
                    'minimum_required': min_required,
                    'buffer_remaining': buffer_remaining,
                    'breach': breach,
                    'scenario': scenario_data['scenario'].iloc[0] if 'scenario' in scenario_data else 'unknown'
                })
        
        return pd.DataFrame(trajectories)
    
    def _calculate_credit_losses(self, cre_exposure: float, 
                                 residential_exposure: float,
                                 consumer_loans: float,
                                 macro: dict) -> float:
        """
        Calculate credit losses based on exposure and macro conditions
        """
        # Loss rates by asset class (scenario-dependent)
        scenario = macro.get('scenario', 'baseline')
        
        # Base loss rates
        loss_rates = {
            'baseline': {'cre': 0.005, 'residential': 0.003, 'consumer': 0.04},
            'adverse': {'cre': 0.04, 'residential': 0.02, 'consumer': 0.08},
            'severely_adverse': {'cre': 0.12, 'residential': 0.06, 'consumer': 0.12}
        }
        
        rates = loss_rates.get(scenario, loss_rates['baseline'])
        
        # Adjust for macro conditions
        unemployment = macro.get('unemployment_rate', 4.5)
        hpi_growth = macro.get('hpi_growth', 0)
        vix = macro.get('vix', 18)
        
        # Unemployment impact on consumer losses
        unemp_factor = 1 + max(0, (unemployment - 4.5) / 10)
        
        # HPI impact on CRE and residential losses
        hpi_factor = 1 + max(0, -hpi_growth / 20)
        
        # VIX impact on CRE losses (market stress)
        vix_factor = 1 + max(0, (vix - 20) / 100)
        
        # Calculate losses (quarterly)
        cre_loss = (cre_exposure * rates['cre'] * hpi_factor * vix_factor) / 4
        residential_loss = (residential_exposure * rates['residential'] * hpi_factor) / 4
        consumer_loss = (consumer_loans * rates['consumer'] * unemp_factor) / 4
        
        return cre_loss + residential_loss + consumer_loss
    
    def _calculate_rwa_growth(self, macro: dict, quarter: int) -> float:
        """Calculate RWA growth factor based on stress conditions"""
        scenario = macro.get('scenario', 'baseline')
        
        # Base RWA growth factors (stress increases risk weights)
        base_growth = {
            'baseline': 1.005,  # 0.5% quarterly growth
            'adverse': 1.015,   # 1.5% quarterly growth
            'severely_adverse': 1.025  # 2.5% quarterly growth
        }
        
        return base_growth.get(scenario, 1.005)
    
    def _calculate_dividends(self, cet1_capital: float, macro: dict) -> float:
        """Calculate dividend payments (reduced in stress)"""
        scenario = macro.get('scenario', 'baseline')
        
        # Dividend rates (annualized, divided by 4 for quarterly)
        div_rates = {
            'baseline': 0.03,  # 3% annual
            'adverse': 0.01,   # 1% annual (cut)
            'severely_adverse': 0.0  # Suspended
        }
        
        return cet1_capital * div_rates.get(scenario, 0.03) / 4
    
    def get_breach_analysis(self, trajectory_df: pd.DataFrame) -> Dict:
        """
        Analyze capital breaches across all banks
        """
        breaches = trajectory_df[trajectory_df['breach'] == True]
        
        analysis = {
            'total_banks': trajectory_df['bank_id'].nunique(),
            'banks_with_breach': breaches['bank_id'].nunique(),
            'breach_rate': breaches['bank_id'].nunique() / max(trajectory_df['bank_id'].nunique(), 1),
            'first_breach_quarter': breaches['quarter'].min() if len(breaches) > 0 else None,
            'worst_breach': {
                'bank_id': trajectory_df.loc[trajectory_df['cet1_ratio'].idxmin(), 'bank_id'] if len(trajectory_df) > 0 else None,
                'min_cet1': trajectory_df['cet1_ratio'].min() if len(trajectory_df) > 0 else None,
                'quarter': trajectory_df.loc[trajectory_df['cet1_ratio'].idxmin(), 'quarter'] if len(trajectory_df) > 0 else None
            },
            'capital_shortfall': (
                trajectory_df[trajectory_df['buffer_remaining'] < 0]['buffer_remaining'].sum()
            )
        }
        
        return analysis
    
    def compare_scenarios(self, bank_data: pd.DataFrame,
                         baseline_trajectory: pd.DataFrame,
                         stress_trajectory: pd.DataFrame) -> pd.DataFrame:
        """Compare capital trajectories across scenarios"""
        
        comparison = []
        
        for bank_id in bank_data['bank_id'].unique():
            baseline = baseline_trajectory[baseline_trajectory['bank_id'] == bank_id]
            stress = stress_trajectory[stress_trajectory['bank_id'] == bank_id]
            
            if len(baseline) > 0 and len(stress) > 0:
                comparison.append({
                    'bank_id': bank_id,
                    'baseline_min_cet1': baseline['cet1_ratio'].min(),
                    'stress_min_cet1': stress['cet1_ratio'].min(),
                    'cet1_decline': baseline['cet1_ratio'].min() - stress['cet1_ratio'].min(),
                    'baseline_breach': baseline['breach'].any(),
                    'stress_breach': stress['breach'].any(),
                    'total_losses_baseline': baseline['credit_losses'].sum(),
                    'total_losses_stress': stress['credit_losses'].sum()
                })
        
        return pd.DataFrame(comparison)
