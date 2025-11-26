"""Time Series Forecasting Package

This package provides models for time series forecasting:
- ARIMA: Statistical forecasting
- LSTM: Deep learning forecasting  
- Prophet: Facebook's forecasting library
"""

__version__ = '1.0.0'
__author__ = 'Amal S P'

from src.models.arima import ARIMAForecaster
from src.models.lstm import LSTMForecaster, LSTMTrainer
from src.models.prophet import ProphetForecaster

__all__ = [
    'ARIMAForecaster',
    'LSTMForecaster',
    'LSTMTrainer',
    'ProphetForecaster'
]
