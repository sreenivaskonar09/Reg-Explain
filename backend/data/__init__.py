"""Data module initialization"""
from .synthetic_generator import SyntheticBankDataGenerator, get_fed_2026_scenarios
from .fred_integration import FREDDataFetcher, FRED_SERIES

__all__ = [
    'SyntheticBankDataGenerator',
    'get_fed_2026_scenarios', 
    'FREDDataFetcher',
    'FRED_SERIES'
]
