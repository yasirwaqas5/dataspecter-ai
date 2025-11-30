"""Simplified Forecast and Anomaly Agents"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from typing import Dict, Optional, List, Tuple


class ForecastAgent:
    """Simple forecasting agent"""
    
    @staticmethod
    def train_and_forecast(daily_df: pd.DataFrame, target_col: str, date_col: str, horizon: int = 14) -> Tuple[Dict, List]:
        """Train model and generate forecast"""
        feats = [c for c in daily_df.columns if c not in [date_col, target_col]]
        X = daily_df[feats].values
        y = daily_df[target_col].values
        
        # Simple train/val split
        split_idx = len(X) - 14
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train simple RF model
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        val_pred = model.predict(X_val)
        # Fix for deprecated 'squared' parameter in mean_squared_error
        mse = mean_squared_error(y_val, val_pred)
        rmse = float(np.sqrt(mse))
        metrics = {
            'rf': {
                'rmse': rmse,
                'mae': float(mean_absolute_error(y_val, val_pred)),
                'r2': float(r2_score(y_val, val_pred)),
                'mape': float(np.mean(np.abs((y_val - val_pred) / (y_val + 1e-9))) * 100)
            }
        }
        
        # Simple forecast (using last value + trend)
        forecast = []
        last_date = daily_df[date_col].max()
        last_val = daily_df[target_col].iloc[-1]
        trend = (daily_df[target_col].iloc[-7:].mean() - daily_df[target_col].iloc[-14:-7].mean())
        
        for i in range(1, horizon + 1):
            next_date = last_date + pd.Timedelta(days=i)
            pred_val = last_val + (trend * i)
            forecast.append({
                'date': str(next_date.date()),
                'prediction': float(max(0, pred_val)),
                'lower_bound': float(max(0, pred_val * 0.9)),
                'upper_bound': float(pred_val * 1.1)
            })
        
        state = {'models': {'rf': model}, 'feats': feats, 'metrics': metrics}
        return state, forecast


class AnomalyAgent:
    """Multi-method anomaly detection"""
    
    @staticmethod
    def detect(df: pd.DataFrame, target_col: str, date_col: str) -> Dict:
        """Detect anomalies using multiple methods"""
        # Z-score method
        z = np.abs((df[target_col] - df[target_col].mean()) / (df[target_col].std() + 1e-9))
        z_anoms = [{'date': str(df[date_col].iloc[i].date()), 'value': float(df[target_col].iloc[i]), 
                    'z_score': float(z.iloc[i]), 'severity': 'HIGH'}
                   for i in np.where(z > 3.0)[0]]
        
        # Isolation Forest
        iso = IsolationForest(contamination=0.02, random_state=42)
        iso_df = df[[target_col]].fillna(0)
        is_anom = iso.fit_predict(iso_df) == -1
        iso_anoms = [{'date': str(df[date_col].iloc[i].date()), 'value': float(df[target_col].iloc[i]), 
                      'severity': 'MEDIUM'}
                     for i in np.where(is_anom)[0]]
        
        # Combine
        all_dates = set([a['date'] for a in z_anoms + iso_anoms])
        ensemble = [{'date': d, 'value': next((a['value'] for a in z_anoms if a['date'] == d), 0),
                    'severity': 'CRITICAL' if d in [a['date'] for a in z_anoms] else 'HIGH'}
                   for d in all_dates]
        
        return {'ensemble': ensemble, 'zscore': z_anoms, 'isolation_forest': iso_anoms}
