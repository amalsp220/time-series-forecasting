import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')

class ARIMAForecaster:
    """ARIMA-based time series forecasting model"""
    
    def __init__(self, order=(1, 1, 1), seasonal_order=None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        
    def auto_tune(self, data, seasonal=False, m=12):
        """Automatically find best ARIMA parameters"""
        self.fitted_model = auto_arima(
            data,
            seasonal=seasonal,
            m=m,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        self.order = self.fitted_model.order
        return self.fitted_model
        
    def fit(self, data):
        """Fit ARIMA model"""
        self.model = ARIMA(data, order=self.order)
        self.fitted_model = self.model.fit()
        return self.fitted_model
        
    def predict(self, steps=30):
        """Generate forecasts"""
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast
        
    def get_summary(self):
        """Get model summary"""
        return self.fitted_model.summary()
        
    def get_residuals(self):
        """Get model residuals"""
        return self.fitted_model.resid
