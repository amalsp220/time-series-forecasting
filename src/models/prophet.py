import pandas as pd
import numpy as np
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

class ProphetForecaster:
    """Facebook Prophet-based time series forecasting model"""
    
    def __init__(self, growth='linear', seasonality_mode='additive'):
        self.model = Prophet(
            growth=growth,
            seasonality_mode=seasonality_mode,
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        self.fitted = False
        
    def add_seasonality(self, name, period, fourier_order):
        """Add custom seasonality"""
        self.model.add_seasonality(
            name=name,
            period=period,
            fourier_order=fourier_order
        )
        
    def add_country_holidays(self, country_name='US'):
        """Add country-specific holidays"""
        self.model.add_country_holidays(country_name=country_name)
        
    def fit(self, data, date_col='ds', value_col='y'):
        """Fit Prophet model
        
        Args:
            data: DataFrame with date and value columns
            date_col: Name of date column (default: 'ds')
            value_col: Name of value column (default: 'y')
        """
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            df = pd.DataFrame({
                'ds': pd.date_range(start='2020-01-01', periods=len(data), freq='D'),
                'y': data
            })
        
        self.model.fit(df)
        self.fitted = True
        return self
        
    def predict(self, periods=30, freq='D'):
        """Generate forecasts
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency of forecasts ('D', 'W', 'M', etc.)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
            
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
    def get_components(self):
        """Get forecast components (trend, seasonality)"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        return self.model.component_modes
        
    def plot_forecast(self, forecast):
        """Plot forecast"""
        from matplotlib import pyplot as plt
        fig = self.model.plot(forecast)
        return fig
        
    def plot_components(self, forecast):
        """Plot forecast components"""
        from matplotlib import pyplot as plt
        fig = self.model.plot_components(forecast)
        return fig
