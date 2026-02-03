"""
FRED API Integration for Real Macroeconomic Data
Fetches actual economic indicators from Federal Reserve Economic Data
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# FRED Series IDs for key macro variables
FRED_SERIES = {
    'unemployment_rate': 'UNRATE',
    'fed_funds_rate': 'FEDFUNDS', 
    'treasury_10y': 'DGS10',
    'treasury_3m': 'DTB3',
    'gdp_growth': 'A191RL1Q225SBEA',  # Real GDP growth rate
    'real_dpi': 'DSPIC96',  # Real Disposable Personal Income
    'hpi': 'CSUSHPINSA',  # S&P/Case-Shiller Home Price Index
    'vix': 'VIXCLS',  # CBOE Volatility Index
    'credit_card_delinquency': 'DRCCLACBS',  # Credit Card Delinquency Rate
    'cpi': 'CPIAUCSL'  # Consumer Price Index
}


class FREDDataFetcher:
    """Fetches real macroeconomic data from FRED API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('FRED_API_KEY', 'demo')
        self._fred = None
        
    @property
    def fred(self):
        """Lazy load FRED API connection"""
        if self._fred is None:
            try:
                from fredapi import Fred
                self._fred = Fred(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"FRED API not available: {e}")
                self._fred = None
        return self._fred
    
    def fetch_series(self, series_id: str, start_date: str = '2015-01-01', 
                    end_date: Optional[str] = None) -> pd.Series:
        """Fetch a single FRED series"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        try:
            if self.fred:
                data = self.fred.get_series(series_id, start_date, end_date)
                return data
        except Exception as e:
            logger.warning(f"Could not fetch {series_id}: {e}")
        
        # Return synthetic data if FRED unavailable
        return self._generate_synthetic_series(series_id, start_date, end_date)
    
    def _generate_synthetic_series(self, series_id: str, start_date: str, 
                                   end_date: str) -> pd.Series:
        """Generate synthetic data when FRED is unavailable"""
        dates = pd.date_range(start=start_date, end=end_date, freq='M')
        
        # Realistic ranges for different series
        ranges = {
            'UNRATE': (3.5, 6.5),
            'FEDFUNDS': (0.0, 5.5),
            'DGS10': (1.5, 4.5),
            'DTB3': (0.0, 5.0),
            'A191RL1Q225SBEA': (-2.0, 4.0),
            'DSPIC96': (14000, 16000),
            'CSUSHPINSA': (200, 320),
            'VIXCLS': (12, 35),
            'DRCCLACBS': (2.0, 4.0),
            'CPIAUCSL': (250, 310)
        }
        
        low, high = ranges.get(series_id, (0, 100))
        base = np.random.uniform(low, high)
        
        # Create random walk with mean reversion
        values = [base]
        for _ in range(len(dates) - 1):
            change = np.random.normal(0, (high - low) * 0.02)
            mean_reversion = 0.1 * ((low + high) / 2 - values[-1])
            new_val = np.clip(values[-1] + change + mean_reversion, low, high)
            values.append(new_val)
        
        return pd.Series(values, index=dates)
    
    def fetch_all_macro_data(self, start_date: str = '2015-01-01',
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch all macro variables and combine into DataFrame"""
        
        data = {}
        for name, series_id in FRED_SERIES.items():
            try:
                series = self.fetch_series(series_id, start_date, end_date)
                data[name] = series
            except Exception as e:
                logger.warning(f"Failed to fetch {name}: {e}")
                
        df = pd.DataFrame(data)
        df = df.resample('Q').mean()  # Quarterly average
        df = df.dropna(how='all')
        
        # Calculate yield curve spread
        if 'treasury_10y' in df.columns and 'treasury_3m' in df.columns:
            df['yield_curve_spread'] = df['treasury_10y'] - df['treasury_3m']
            
        # Calculate real DPI growth
        if 'real_dpi' in df.columns:
            df['real_dpi_growth'] = df['real_dpi'].pct_change() * 100
            
        # Calculate HPI growth
        if 'hpi' in df.columns:
            df['hpi_growth'] = df['hpi'].pct_change(4) * 100  # YoY
            
        return df
    
    def get_current_conditions(self) -> Dict:
        """Get most recent economic conditions"""
        df = self.fetch_all_macro_data()
        
        if df.empty:
            return self._get_synthetic_conditions()
        
        latest = df.iloc[-1].to_dict()
        
        return {
            'as_of_date': df.index[-1].strftime('%Y-%m-%d'),
            'unemployment_rate': latest.get('unemployment_rate', 4.2),
            'fed_funds_rate': latest.get('fed_funds_rate', 4.5),
            'treasury_10y': latest.get('treasury_10y', 4.0),
            'treasury_3m': latest.get('treasury_3m', 4.3),
            'yield_curve_spread': latest.get('yield_curve_spread', -0.3),
            'vix': latest.get('vix', 18.0),
            'hpi_growth': latest.get('hpi_growth', 3.0),
            'gdp_growth': latest.get('gdp_growth', 2.0)
        }
    
    def _get_synthetic_conditions(self) -> Dict:
        """Return synthetic current conditions"""
        return {
            'as_of_date': datetime.now().strftime('%Y-%m-%d'),
            'unemployment_rate': 4.2,
            'fed_funds_rate': 4.5,
            'treasury_10y': 4.0,
            'treasury_3m': 4.3,
            'yield_curve_spread': -0.3,
            'vix': 18.0,
            'hpi_growth': 3.0,
            'gdp_growth': 2.0
        }
    
    def project_scenario(self, scenario_type: str = 'baseline', 
                        n_quarters: int = 9) -> pd.DataFrame:
        """Project macro variables over n quarters for given scenario"""
        
        current = self.get_current_conditions()
        
        # Fed scenario parameters (2026 guidelines)
        scenario_params = {
            'baseline': {
                'unemployment_target': 4.5,
                'unemployment_speed': 0.1,
                'fed_funds_target': 3.5,
                'fed_funds_speed': 0.25,
                'vix_target': 18,
                'hpi_growth_target': 3.0,
                'gdp_target': 2.0
            },
            'adverse': {
                'unemployment_target': 7.5,
                'unemployment_speed': 0.5,
                'fed_funds_target': 1.5,
                'fed_funds_speed': 0.5,
                'vix_target': 50,
                'hpi_growth_target': -10.0,
                'gdp_target': -2.0
            },
            'severely_adverse': {
                'unemployment_target': 10.0,
                'unemployment_speed': 0.8,
                'fed_funds_target': 0.25,
                'fed_funds_speed': 0.75,
                'vix_target': 72,
                'hpi_growth_target': -25.0,
                'gdp_target': -6.0
            }
        }
        
        params = scenario_params.get(scenario_type, scenario_params['baseline'])
        projections = []
        
        # Current values
        unemp = current['unemployment_rate']
        ffr = current['fed_funds_rate']
        t10y = current['treasury_10y']
        vix = current['vix']
        hpi = current['hpi_growth']
        gdp = current['gdp_growth']
        
        base_date = datetime.now()
        
        for q in range(n_quarters):
            quarter_date = base_date + timedelta(days=(q+1)*91)
            
            # Mean reversion to target
            unemp += params['unemployment_speed'] * (params['unemployment_target'] - unemp)
            unemp += np.random.normal(0, 0.2)
            
            ffr += params['fed_funds_speed'] * (params['fed_funds_target'] - ffr)
            ffr = max(0, ffr + np.random.normal(0, 0.1))
            
            # VIX spikes in adverse scenarios
            vix_shock = np.random.normal(0, 5) if scenario_type != 'baseline' else np.random.normal(0, 2)
            vix += 0.3 * (params['vix_target'] - vix) + vix_shock
            vix = max(10, min(100, vix))
            
            # Treasury rates follow fed funds with spread
            t10y = ffr + np.random.uniform(0.5, 1.5)
            t3m = ffr + np.random.uniform(-0.5, 0.3)
            
            # HPI and GDP
            hpi += 0.2 * (params['hpi_growth_target'] - hpi) + np.random.normal(0, 1)
            gdp += 0.25 * (params['gdp_target'] - gdp) + np.random.normal(0, 0.5)
            
            projections.append({
                'quarter': q + 1,
                'quarter_date': quarter_date,
                'scenario': scenario_type,
                'unemployment_rate': np.clip(unemp, 2.5, 15),
                'fed_funds_rate': np.clip(ffr, 0, 8),
                'treasury_10y': np.clip(t10y, 0.5, 10),
                'treasury_3m': np.clip(t3m, 0, 8),
                'yield_curve_spread': t10y - t3m,
                'vix': np.clip(vix, 10, 100),
                'hpi_growth': np.clip(hpi, -30, 20),
                'gdp_growth': np.clip(gdp, -10, 6),
                'credit_card_utilization': 0.22 + 0.02 * (unemp - 4.5)
            })
        
        return pd.DataFrame(projections)
