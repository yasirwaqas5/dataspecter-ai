"""
Advanced Time-Series Processor
Implements:
- Automatic date detection
- Seasonality detection (STL decomposition)
- Trend analysis
- YOY/MOM calculations
- Missing data interpolation
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


class TimeSeriesProcessor:
    """Advanced time-series preprocessing and analysis"""
    
    def __init__(self):
        self.seasonal_period = None
        self.decomposition_results = None
        
    @staticmethod
    def auto_detect_date_column(df: pd.DataFrame) -> Optional[str]:
        """
        Automatically detect the date column in a DataFrame
        
        Returns:
            Column name or None if not found
        """
        # Strategy 1: Check column names
        date_keywords = ['date', 'time', 'datetime', 'timestamp', 'day', 'month', 'year', 'period']
        
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in date_keywords):
                # Try to parse as datetime
                try:
                    pd.to_datetime(df[col].dropna().head(100))
                    return col
                except:
                    continue
        
        # Strategy 2: Check each column's data
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Try parsing a sample
                    sample = df[col].dropna().head(100)
                    parsed = pd.to_datetime(sample, errors='coerce')
                    
                    # If >80% successfully parsed, it's likely a date column
                    if parsed.notna().sum() / len(sample) > 0.8:
                        return col
                except:
                    continue
        
        return None
    
    @staticmethod
    def prepare_timeseries(df: pd.DataFrame, date_col: str, value_col: str, 
                          freq: str = 'D') -> pd.DataFrame:
        """
        Prepare time-series data with proper indexing and frequency
        
        Args:
            df: Input DataFrame
            date_col: Name of date column
            value_col: Name of value column
            freq: Frequency ('D', 'W', 'M', 'Q', 'Y')
            
        Returns:
            Resampled time-series DataFrame
        """
        ts_df = df[[date_col, value_col]].copy()
        
        # Convert to datetime
        ts_df[date_col] = pd.to_datetime(ts_df[date_col])
        
        # Remove duplicates, keep first
        ts_df = ts_df.drop_duplicates(subset=[date_col], keep='first')
        
        # Set index
        ts_df = ts_df.set_index(date_col).sort_index()
        
        # Resample and aggregate
        ts_df = ts_df.resample(freq).sum()
        
        # Reset index
        ts_df = ts_df.reset_index()
        ts_df.columns = [date_col, value_col]
        
        return ts_df
    
    @staticmethod
    def fill_missing_dates(df: pd.DataFrame, date_col: str, value_col: str,
                          method: str = 'interpolate') -> pd.DataFrame:
        """
        Fill missing dates in time-series
        
        Args:
            df: Input DataFrame
            date_col: Name of date column
            value_col: Name of value column
            method: 'interpolate', 'ffill', 'bfill', 'zero'
            
        Returns:
            DataFrame with filled dates
        """
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        df_copy = df_copy.set_index(date_col).sort_index()
        
        # Create full date range
        full_range = pd.date_range(
            start=df_copy.index.min(),
            end=df_copy.index.max(),
            freq='D'
        )
        
        # Reindex
        df_copy = df_copy.reindex(full_range)
        
        # Fill missing values
        if method == 'interpolate':
            df_copy[value_col] = df_copy[value_col].interpolate(method='linear')
        elif method == 'ffill':
            df_copy[value_col] = df_copy[value_col].fillna(method='ffill')
        elif method == 'bfill':
            df_copy[value_col] = df_copy[value_col].fillna(method='bfill')
        elif method == 'zero':
            df_copy[value_col] = df_copy[value_col].fillna(0)
        
        df_copy = df_copy.reset_index()
        df_copy.columns = [date_col, value_col]
        
        return df_copy
    
    @staticmethod
    def detect_seasonality(series: pd.Series, max_lag: int = 365) -> Dict:
        """
        Detect seasonality using autocorrelation
        
        Returns:
            Dict with seasonality information
        """
        from statsmodels.tsa.stattools import acf
        
        result = {
            'has_seasonality': False,
            'period': None,
            'strength': 0.0,
            'periods_detected': []
        }
        
        try:
            # Calculate autocorrelation
            max_lag = min(max_lag, len(series) // 2)
            autocorr = acf(series.dropna(), nlags=max_lag, fft=True)
            
            # Find peaks in autocorrelation
            peaks = []
            for i in range(7, len(autocorr) - 1):  # Start from lag 7 (weekly)
                if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    if autocorr[i] > 0.3:  # Significant correlation
                        peaks.append((i, autocorr[i]))
            
            # Check for common seasonal periods
            common_periods = {
                7: 'weekly',
                30: 'monthly',
                91: 'quarterly',
                365: 'yearly'
            }
            
            for lag, corr in peaks:
                for period, name in common_periods.items():
                    if abs(lag - period) <= 2:  # Allow ±2 day tolerance
                        result['periods_detected'].append({
                            'period': period,
                            'name': name,
                            'strength': float(corr)
                        })
            
            if result['periods_detected']:
                result['has_seasonality'] = True
                # Use strongest period
                strongest = max(result['periods_detected'], key=lambda x: x['strength'])
                result['period'] = strongest['period']
                result['strength'] = strongest['strength']
        
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    @staticmethod
    def detect_seasonality_fft(series: pd.Series) -> Dict:
        """
        Detect seasonality using FFT (Fast Fourier Transform)
        
        Returns:
            Dict with seasonality information
        """
        from scipy import signal
        import numpy as np
        
        result = {
            'has_seasonality': False,
            'period': None,
            'strength': 0.0,
            'method': 'fft'
        }
        
        try:
            # Remove NaN values
            clean_series = series.dropna()
            if len(clean_series) < 10:
                result['error'] = "Insufficient data for FFT analysis"
                return result
            
            # Compute periodogram
            frequencies, power = signal.periodogram(clean_series.values)
            
            # Handle case where all power values are zero or NaN
            if np.all(power == 0) or np.all(np.isnan(power)):
                result['period'] = 12  # Default period
                result['strength'] = 0.0
                return result
            
            # Find peak frequency
            peak_idx = np.argmax(power)
            peak_freq = frequencies[peak_idx]
            peak_power = power[peak_idx]
            
            # Calculate period (avoid division by zero)
            if peak_freq > 0:
                period = int(1 / peak_freq)
            else:
                period = 12  # Default period
            
            # Ensure period is reasonable
            if period <= 1 or period > len(clean_series):
                period = min(12, len(clean_series) // 2)
            
            result['has_seasonality'] = True
            result['period'] = period
            result['strength'] = float(peak_power) if not np.isnan(peak_power) else 0.0
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    @staticmethod
    def detect_seasonality_combined(series: pd.Series) -> Dict:
        """
        Combined approach for seasonality detection using both autocorrelation and FFT
        
        Returns:
            Dict with seasonality information
        """
        import numpy as np
        
        result = {
            'has_seasonality': False,
            'period': None,
            'strength': 0.0,
            'methods': {}
        }
        
        try:
            # Method 1: Autocorrelation
            acf_result = TimeSeriesProcessor.detect_seasonality(series)
            result['methods']['acf'] = acf_result
            
            # Method 2: FFT
            fft_result = TimeSeriesProcessor.detect_seasonality_fft(series)
            result['methods']['fft'] = fft_result
            
            # Combine results
            if acf_result.get('has_seasonality') and fft_result.get('has_seasonality'):
                # Both methods detected seasonality, use the stronger one
                if acf_result.get('strength', 0) > fft_result.get('strength', 0):
                    result['has_seasonality'] = True
                    result['period'] = acf_result.get('period')
                    result['strength'] = acf_result.get('strength')
                else:
                    result['has_seasonality'] = True
                    result['period'] = fft_result.get('period')
                    result['strength'] = fft_result.get('strength')
            elif acf_result.get('has_seasonality'):
                result['has_seasonality'] = True
                result['period'] = acf_result.get('period')
                result['strength'] = acf_result.get('strength')
            elif fft_result.get('has_seasonality'):
                result['has_seasonality'] = True
                result['period'] = fft_result.get('period')
                result['strength'] = fft_result.get('strength')
            else:
                # No seasonality detected, but provide a default
                result['period'] = 12  # Default period
                
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    @staticmethod
    def decompose_timeseries(df: pd.DataFrame, date_col: str, value_col: str,
                            period: Optional[int] = None) -> Dict:
        """
        Decompose time-series into trend, seasonal, and residual components
        Uses STL (Seasonal and Trend decomposition using Loess)
        
        Returns:
            Dict with decomposition components
        """
        from statsmodels.tsa.seasonal import STL
        
        result = {
            'success': False,
            'trend': None,
            'seasonal': None,
            'residual': None,
            'period': period
        }
        
        try:
            # Prepare series
            ts = df.set_index(date_col)[value_col]
            
            # Auto-detect period if not provided
            if period is None:
                seasonality_info = TimeSeriesProcessor.detect_seasonality(ts)
                period = seasonality_info.get('period', 7)  # Default to weekly
            
            # Minimum data requirement
            if len(ts) < 2 * period:
                result['error'] = f"Need at least {2 * period} observations for period {period}"
                return result
            
            # Perform STL decomposition
            stl = STL(ts.fillna(method='ffill'), seasonal=period, robust=True)
            decomposition = stl.fit()
            
            # Extract components
            result['trend'] = decomposition.trend.reset_index()
            result['trend'].columns = [date_col, 'trend']
            
            result['seasonal'] = decomposition.seasonal.reset_index()
            result['seasonal'].columns = [date_col, 'seasonal']
            
            result['residual'] = decomposition.resid.reset_index()
            result['residual'].columns = [date_col, 'residual']
            
            # Calculate strength of components
            var_residual = np.var(decomposition.resid)
            var_detrend = np.var(decomposition.seasonal + decomposition.resid)
            
            result['seasonal_strength'] = max(0, 1 - (var_residual / var_detrend)) if var_detrend > 0 else 0
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    @staticmethod
    def calculate_growth_metrics(df: pd.DataFrame, date_col: str, value_col: str) -> Dict:
        """
        Calculate YOY (Year-over-Year) and MOM (Month-over-Month) growth
        
        Returns:
            Dict with growth metrics
        """
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        df_copy = df_copy.sort_values(date_col)
        
        result = {
            'yoy_growth': [],
            'mom_growth': [],
            'avg_yoy_growth': 0.0,
            'avg_mom_growth': 0.0
        }
        
        try:
            # Add year and month columns
            df_copy['year'] = df_copy[date_col].dt.year
            df_copy['month'] = df_copy[date_col].dt.month
            
            # Aggregate by month
            monthly = df_copy.groupby([df_copy[date_col].dt.to_period('M')])[value_col].sum()
            monthly_df = monthly.reset_index()
            monthly_df.columns = ['period', value_col]
            monthly_df['date'] = monthly_df['period'].dt.to_timestamp()
            
            # Calculate MOM
            monthly_df['mom_growth'] = monthly_df[value_col].pct_change() * 100
            
            result['mom_growth'] = monthly_df[['date', 'mom_growth']].dropna().to_dict('records')
            result['avg_mom_growth'] = float(monthly_df['mom_growth'].mean()) if not monthly_df['mom_growth'].isna().all() else 0.0
            
            # Calculate YOY
            yearly = df_copy.groupby('year')[value_col].sum()
            yearly_df = yearly.reset_index()
            yearly_df['yoy_growth'] = yearly_df[value_col].pct_change() * 100
            
            result['yoy_growth'] = yearly_df[['year', 'yoy_growth']].dropna().to_dict('records')
            result['avg_yoy_growth'] = float(yearly_df['yoy_growth'].mean()) if not yearly_df['yoy_growth'].isna().all() else 0.0
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    @staticmethod
    def compute_rolling_statistics(df: pd.DataFrame, date_col: str, value_col: str,
                                   windows: List[int] = [7, 30, 90]) -> pd.DataFrame:
        """
        Compute rolling statistics for multiple windows
        
        Returns:
            DataFrame with rolling statistics
        """
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        df_copy = df_copy.sort_values(date_col)
        
        for window in windows:
            df_copy[f'rolling_mean_{window}'] = df_copy[value_col].rolling(window=window, min_periods=1).mean()
            df_copy[f'rolling_std_{window}'] = df_copy[value_col].rolling(window=window, min_periods=1).std()
            df_copy[f'rolling_min_{window}'] = df_copy[value_col].rolling(window=window, min_periods=1).min()
            df_copy[f'rolling_max_{window}'] = df_copy[value_col].rolling(window=window, min_periods=1).max()
        
        return df_copy
