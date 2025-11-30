"""
Advanced Forecasting Engine
Implements:
- ARIMA with auto parameter selection
- Facebook Prophet
- Random Forest (existing)
- Automatic model selection based on AIC/performance
- Ensemble forecasting
- Produces Plotly figures for UI (forecast plot, Prophet components, ARIMA ACF/PACF)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Plotly (for visual outputs)
import plotly.graph_objects as go
import plotly.express as px

class AdvancedForecastAgent:
    """Multi-model forecasting with automatic selection and visual outputs"""
    
    def __init__(self):
        self.models = {}
        self.best_model_name = None
        self.performance_metrics = {}
    
    @staticmethod
    def _ensure_datetime_df(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        return df.sort_values(date_col).reset_index(drop=True)
    
    @staticmethod
    def train_arima(ts_data: pd.Series, horizon: int = 14, 
                    seasonal: bool = False, m: int = 7) -> Tuple[Dict, List]:
        """
        Train ARIMA model with automatic parameter selection
        
        Returns:
            (model_info, forecast_list)
        """
        try:
            from statsmodels.tsa.stattools import adfuller
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf
        except Exception:
            return {
                'success': False,
                'error': 'statsmodels not installed. Run: pip install statsmodels',
            }, []
        
        result = {
            'success': False,
            'model_type': 'SARIMA' if seasonal else 'ARIMA',
            'params': {},
            'aic': None,
            'bic': None,
            'metrics': {},
            # placeholders for diagnostics/plots
            'acf_fig': None,
            'pacf_fig': None,
        }
        
        try:
            # ensure ts_data is sorted and has datetime index if possible
            if isinstance(ts_data.index, pd.DatetimeIndex):
                ts = ts_data.dropna().sort_index()
            else:
                # attempt to coerce index to DatetimeIndex if values are datetime-like
                ts = ts_data.dropna()
                try:
                    ts.index = pd.to_datetime(ts.index)
                    ts = ts.sort_index()
                except Exception:
                    ts = ts  # keep as-is
            
            # Check stationarity (ADF)
            try:
                adf_result = adfuller(ts)
                is_stationary = adf_result[1] < 0.05
                d = 0 if is_stationary else 1
            except Exception:
                # fallback
                d = 0
            
            # Auto parameter search (simplified grid)
            best_aic = np.inf
            best_order = (1, d, 1)
            best_seasonal_order = (1, 1, 1, m) if seasonal else None
            
            p_range = [0, 1, 2]
            q_range = [0, 1, 2]
            
            for p in p_range:
                for q in q_range:
                    try:
                        if seasonal:
                            temp_model = SARIMAX(
                                ts,
                                order=(p, d, q),
                                seasonal_order=(1, 1, 1, m),
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                        else:
                            temp_model = ARIMA(ts, order=(p, d, q))
                        
                        temp_fit = temp_model.fit(disp=False)
                        if temp_fit.aic < best_aic:
                            best_aic = temp_fit.aic
                            best_order = (p, d, q)
                            if seasonal:
                                best_seasonal_order = (1, 1, 1, m)
                    except Exception:
                        continue
            
            # Train final model
            if seasonal:
                model = SARIMAX(
                    ts,
                    order=best_order,
                    seasonal_order=best_seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                model = ARIMA(ts, order=best_order)
            
            fitted_model = model.fit(disp=False)
            
            # Forecast
            forecast_obj = fitted_model.get_forecast(steps=horizon)
            forecast_values = forecast_obj.predicted_mean
            conf_int = forecast_obj.conf_int()
            
            # Compose forecast_list
            # If ts has DatetimeIndex we'll increment that, else use daily increments from last index
            try:
                last_date = ts.index[-1]
                if not isinstance(last_date, pd.Timestamp):
                    last_date = pd.Timestamp(last_date)
            except Exception:
                last_date = pd.Timestamp.now()
            
            forecast_list = []
            for i in range(horizon):
                next_date = last_date + pd.Timedelta(days=i+1)
                # ensure bounds exist
                l = conf_int.iloc[i, 0] if conf_int.shape[1] >= 2 else forecast_values[i] * 0.9
                u = conf_int.iloc[i, 1] if conf_int.shape[1] >= 2 else forecast_values[i] * 1.1
                forecast_list.append({
                    'date': str(pd.Timestamp(next_date).date()),
                    'prediction': float(max(0.0, float(forecast_values[i]))),
                    'lower_bound': float(max(0.0, float(l))),
                    'upper_bound': float(max(0.0, float(u)))
                })
            
            # residuals & metrics
            residuals = fitted_model.resid
            rmse = float(np.sqrt(np.mean((residuals) ** 2)))
            mae = float(np.mean(np.abs(residuals)))
            mape = float(np.mean(np.abs(residuals / (ts + 1e-9))) * 100)
            
            result.update({
                'success': True,
                'params': {
                    'order': best_order,
                    'seasonal_order': best_seasonal_order if seasonal else None
                },
                'aic': float(getattr(fitted_model, 'aic', np.nan)),
                'bic': float(getattr(fitted_model, 'bic', np.nan)),
                'metrics': {
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape
                }
            })
            
            # Generate ACF/PACF diagnostics as Plotly Figures (if possible)
            try:
                from statsmodels.tsa.stattools import acf as sm_acf, pacf as sm_pacf
                acf_vals = sm_acf(residuals, nlags=min(40, len(residuals)-1), fft=False)
                pacf_vals = sm_pacf(residuals, nlags=min(40, len(residuals)-1))
                lags = list(range(len(acf_vals)))
                
                acf_fig = go.Figure()
                acf_fig.add_trace(go.Bar(x=lags, y=acf_vals, name='ACF'))
                acf_fig.update_layout(title='Residuals ACF', xaxis_title='Lag', yaxis_title='ACF')
                
                pacf_fig = go.Figure()
                pacf_fig.add_trace(go.Bar(x=lags, y=pacf_vals, name='PACF'))
                pacf_fig.update_layout(title='Residuals PACF', xaxis_title='Lag', yaxis_title='PACF')
                
                result['acf_fig'] = acf_fig
                result['pacf_fig'] = pacf_fig
            except Exception:
                result['acf_fig'] = None
                result['pacf_fig'] = None
            
            return result, forecast_list
            
        except Exception as e:
            result['error'] = str(e)
            return result, []
    
    @staticmethod
    def train_prophet(df: pd.DataFrame, date_col: str, value_col: str,
                     horizon: int = 14, seasonality_mode: str = 'additive') -> Tuple[Dict, List]:
        """
        Train Prophet model and produce component plots
        """
        try:
            from prophet import Prophet
        except Exception:
            return {
                'success': False,
                'error': 'Prophet not installed. Run: pip install prophet'
            }, []
        
        result = {
            'success': False,
            'model_type': 'Prophet',
            'params': {'seasonality_mode': seasonality_mode},
            'metrics': {},
            'prophet_components': {},  # will hold Plotly figures
        }
        
        try:
            prophet_df = df[[date_col, value_col]].copy()
            prophet_df.columns = ['ds', 'y']
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
            prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
            
            model = Prophet(
                seasonality_mode=seasonality_mode,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=len(prophet_df) > 365,
                changepoint_prior_scale=0.05
            )
            
            model.fit(prophet_df)
            
            future = model.make_future_dataframe(periods=horizon, freq='D')
            forecast = model.predict(future)
            forecast_future = forecast.tail(horizon)
            
            forecast_list = []
            for idx, row in forecast_future.iterrows():
                forecast_list.append({
                    'date': str(pd.Timestamp(row['ds']).date()),
                    'prediction': float(max(0.0, float(row.get('yhat', 0.0)))),
                    'lower_bound': float(max(0.0, float(row.get('yhat_lower', row.get('yhat', 0.0)*0.9)))),
                    'upper_bound': float(max(0.0, float(row.get('yhat_upper', row.get('yhat', 0.0)*1.1))))
                })
            
            # training residuals metrics: compare prophet_df.y vs forecast yhat for training range
            forecast_train = forecast.loc[forecast['ds'].isin(prophet_df['ds'])]
            # Align lengths
            try:
                merged = pd.merge(prophet_df, forecast_train[['ds','yhat']], on='ds', how='left')
                residuals = merged['y'] - merged['yhat']
                rmse = float(np.sqrt(np.nanmean(residuals**2)))
                mae = float(np.nanmean(np.abs(residuals)))
                mape = float(np.nanmean(np.abs(residuals / (merged['y'] + 1e-9))) * 100)
            except Exception:
                rmse = mae = mape = float('nan')
            
            result.update({
                'success': True,
                'metrics': {
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape
                }
            })
            
            # Build Prophet component plots (trend, weekly, yearly) as Plotly figs
            try:
                comp_figs = {}
                # Add the model and forecast to results for later use
                result['prophet_model'] = model
                result['prophet_forecast'] = forecast
                
                # Extract components properly from the Prophet model
                # Prophet's plot_components creates matplotlib figures, but we want Plotly
                # Let's create our own component plots using the forecast data
                
                # Trend component
                if 'trend' in forecast.columns:
                    fig_trend = px.line(forecast, x='ds', y='trend', title='Trend Component')
                    comp_figs['trend'] = fig_trend
                
                # Seasonal components - Prophet adds these to the forecast
                # Check for seasonal components that might be in the forecast
                seasonal_cols = [col for col in forecast.columns if col in ['yearly', 'weekly', 'daily']]
                for seasonal_col in seasonal_cols:
                    if seasonal_col in forecast.columns:
                        fig_seasonal = px.line(forecast, x='ds', y=seasonal_col, title=f'{seasonal_col.capitalize()} Seasonality')
                        comp_figs[seasonal_col] = fig_seasonal
                
                # If we have the model, we can also extract seasonalities that were fitted
                if hasattr(model, 'seasonalities'):
                    for seasonality_name in model.seasonalities.keys():
                        # We can't directly plot these without the model's plot_components,
                        # but we can note that they exist
                        pass
                
                result['prophet_components'] = comp_figs
            except Exception as e:
                # If we can't create Plotly components, store the model for matplotlib fallback
                result['prophet_components'] = {}
                result['prophet_model'] = model
                result['prophet_forecast'] = forecast
            
            return result, forecast_list
        except Exception as e:
            result['error'] = str(e)
            return result, []
    
    @staticmethod
    def train_random_forest(df: pd.DataFrame, target_col: str, date_col: str,
                           horizon: int = 14) -> Tuple[Dict, List]:
        """
        Train Random Forest model with lag features (keeps original simple forecast)
        """
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        except Exception:
            return {
                'success': False,
                'error': 'sklearn not installed. Run: pip install scikit-learn'
            }, []
        
        result = {
            'success': False,
            'model_type': 'Random Forest',
            'params': {},
            'metrics': {}
        }
        
        try:
            # prepare dates
            df_local = df.copy()
            df_local[date_col] = pd.to_datetime(df_local[date_col])
            df_local = df_local.sort_values(date_col).reset_index(drop=True)
            
            # simple lag features: last 7 days and last 14 days means if available
            # But original expects explicit feature columns; follow the original logic:
            feature_cols = [c for c in df_local.columns if c not in [date_col, target_col]]
            
            # if no feature cols, attempt to create lag features
            if not feature_cols:
                # create lag features
                for lag in range(1, 8):
                    df_local[f'lag_{lag}'] = df_local[target_col].shift(lag)
                feature_cols = [c for c in df_local.columns if c not in [date_col, target_col]]
            
            df_local = df_local.dropna().reset_index(drop=True)
            if df_local.shape[0] < 21:
                return {
                    'success': False,
                    'error': 'Not enough data after lag creation for Random Forest (need >=21 rows)'
                }, []
            
            X = df_local[feature_cols].fillna(0).values
            y = df_local[target_col].fillna(0).values
            
            split_idx = max(1, len(X) - 14)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            val_pred = model.predict(X_val)
            
            rmse = float(np.sqrt(mean_squared_error(y_val, val_pred))) if len(y_val) > 0 else float('nan')
            mae = float(mean_absolute_error(y_val, val_pred)) if len(y_val) > 0 else float('nan')
            r2 = float(r2_score(y_val, val_pred)) if len(y_val) > 0 else float('nan')
            mape = float(np.mean(np.abs((y_val - val_pred) / (y_val + 1e-9))) * 100) if len(y_val) > 0 else float('nan')
            
            result.update({
                'success': True,
                'params': {
                    'n_estimators': 200,
                    'max_depth': 15
                },
                'metrics': {
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'mape': mape
                }
            })
            
            # Forecast: simple trend extrapolation as before
            forecast_list = []
            last_date = pd.to_datetime(df[date_col].max())
            last_val = float(df[target_col].iloc[-1])
            # compute simple recent-week trend safely
            try:
                recent7 = df[target_col].iloc[-7:].mean()
                prev7 = df[target_col].iloc[-14:-7].mean()
                trend = recent7 - prev7
            except Exception:
                trend = 0.0
            
            for i in range(1, horizon + 1):
                next_date = last_date + pd.Timedelta(days=i)
                pred_val = last_val + (trend * i)
                forecast_list.append({
                    'date': str(pd.Timestamp(next_date).date()),
                    'prediction': float(max(0.0, pred_val)),
                    'lower_bound': float(max(0.0, pred_val * 0.85)),
                    'upper_bound': float(pred_val * 1.15)
                })
            
            return result, forecast_list
        except Exception as e:
            result['error'] = str(e)
            return result, []
    
    @staticmethod
    def auto_select_best_model(models_results: Dict[str, Tuple[Dict, List]]) -> Tuple[Optional[str], Dict, List]:
        """
        Automatically select best model based on AIC (for ARIMA) or RMSE
        """
        best_model = None
        best_score = np.inf
        best_forecast = []
        best_info = {}
        
        for model_name, (info, forecast) in models_results.items():
            if not info.get('success', False):
                continue
            
            # Prefer lower AIC for statistical models, else RMSE
            score = None
            if info.get('aic') is not None and np.isfinite(info.get('aic')):
                score = info.get('aic')
            elif 'metrics' in info and isinstance(info['metrics'], dict) and info['metrics'].get('rmse') is not None:
                score = info['metrics']['rmse']
            
            if score is None:
                continue
            
            if score < best_score:
                best_score = score
                best_model = model_name
                best_forecast = forecast
                best_info = info
        
        return best_model, best_info, best_forecast
    
    @staticmethod
    def ensemble_forecast(forecasts: Dict[str, List], weights: Optional[Dict[str, float]] = None) -> List:
        """
        Create ensemble forecast by averaging multiple model predictions
        """
        if not forecasts:
            return []
        
        if weights is None:
            weights = {name: 1.0 / len(forecasts) for name in forecasts.keys()}
        
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        horizon = len(next(iter(forecasts.values())))
        ensemble = []
        for i in range(horizon):
            weighted_pred = 0.0
            weighted_lower = 0.0
            weighted_upper = 0.0
            date = None
            for model_name, forecast_list in forecasts.items():
                if i < len(forecast_list):
                    weight = weights.get(model_name, 0.0)
                    weighted_pred += forecast_list[i]['prediction'] * weight
                    weighted_lower += forecast_list[i]['lower_bound'] * weight
                    weighted_upper += forecast_list[i]['upper_bound'] * weight
                    date = forecast_list[i]['date']
            ensemble.append({
                'date': date,
                'prediction': float(weighted_pred),
                'lower_bound': float(weighted_lower),
                'upper_bound': float(weighted_upper)
            })
        return ensemble
    
    @staticmethod
    def train_all_models(df: pd.DataFrame, date_col: str, value_col: str,
                        horizon: int = 14, enable_prophet: bool = True,
                        enable_arima: bool = True) -> Dict:
        """
        Train all available models and return results (includes Plotly figs)
        """
        results = {
            'models': {},
            'best_model': None,
            'best_forecast': [],
            'ensemble_forecast': [],
            'performance_comparison': [],
            # visualization placeholders
            'forecast_plot': None,
            'prophet_components': {},
            'arima_diagnostics': {}
        }
        
        # prepare dataframe: ensure date column is datetime and sorted
        try:
            df_local = AdvancedForecastAgent._ensure_datetime_df(df, date_col)
        except Exception:
            df_local = df.copy()
        
        # Train Random Forest
        try:
            rf_info, rf_forecast = AdvancedForecastAgent.train_random_forest(df_local, value_col, date_col, horizon)
            if rf_info.get('success'):
                results['models']['Random Forest'] = (rf_info, rf_forecast)
        except Exception as e:
            results.setdefault('errors', []).append(f"RF train error: {str(e)}")
        
        # Train ARIMA / SARIMA
        if enable_arima:
            try:
                ts_series = df_local.set_index(date_col)[value_col]
                arima_info, arima_forecast = AdvancedForecastAgent.train_arima(ts_series, horizon, seasonal=False)
                if arima_info.get('success'):
                    results['models']['ARIMA'] = (arima_info, arima_forecast)
                    # attach diagnostics if available
                    if arima_info.get('acf_fig') is not None:
                        results['arima_diagnostics']['ARIMA'] = {
                            'acf_fig': arima_info.get('acf_fig'),
                            'pacf_fig': arima_info.get('pacf_fig')
                        }
                if len(ts_series) > 60:
                    sarima_info, sarima_forecast = AdvancedForecastAgent.train_arima(ts_series, horizon, seasonal=True, m=7)
                    if sarima_info.get('success'):
                        results['models']['SARIMA'] = (sarima_info, sarima_forecast)
                        if sarima_info.get('acf_fig') is not None:
                            results['arima_diagnostics']['SARIMA'] = {
                                'acf_fig': sarima_info.get('acf_fig'),
                                'pacf_fig': sarima_info.get('pacf_fig')
                            }
            except Exception as e:
                results.setdefault('errors', []).append(f"ARIMA train error: {str(e)}")
        
        # Train Prophet
        if enable_prophet:
            try:
                prophet_info, prophet_forecast = AdvancedForecastAgent.train_prophet(df_local, date_col, value_col, horizon)
                if prophet_info.get('success'):
                    results['models']['Prophet'] = (prophet_info, prophet_forecast)
                    # attach components figures if present
                    if isinstance(prophet_info.get('prophet_components', {}), dict):
                        results['prophet_components'] = prophet_info.get('prophet_components', {})
                    # Store Prophet model and forecast for components breakdown
                    if 'prophet_model' in prophet_info:
                        results['prophet_model'] = prophet_info['prophet_model']
                    if 'prophet_forecast' in prophet_info:
                        results['prophet_forecast'] = prophet_info['prophet_forecast']
            except Exception as e:
                results.setdefault('errors', []).append(f"Prophet train error: {str(e)}")
        
        # Select best model
        if results['models']:
            best_name, best_info, best_forecast = AdvancedForecastAgent.auto_select_best_model(results['models'])
            results['best_model'] = best_name
            results['best_forecast'] = best_forecast
            results['best_model_info'] = best_info if best_info else {}
            # Add model_used for display in UI
            results['model_used'] = best_name if best_name else 'N/A'
            
            # Ensemble
            forecasts_dict = {name: forecast for name, (info, forecast) in results['models'].items()}
            results['ensemble_forecast'] = AdvancedForecastAgent.ensemble_forecast(forecasts_dict)
            
            # Performance comparison
            for name, (info, _) in results['models'].items():
                if 'metrics' in info:
                    results['performance_comparison'].append({
                        'model': name,
                        **info.get('metrics', {}),
                        'aic': info.get('aic', None)
                    })
        
        # ==== Create a unified forecast Plotly figure (historical + best forecast + ensemble) ====
        try:
            # build historical series
            hist_df = df_local[[date_col, value_col]].copy()
            hist_df.columns = ['date', 'value']
            hist_df['date'] = pd.to_datetime(hist_df['date'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist_df['date'], y=hist_df['value'],
                                     mode='lines', name='Historical', line=dict(width=2)))
            
            # plot best forecast if exists
            if results.get('best_forecast'):
                best_fc = pd.DataFrame(results['best_forecast'])
                best_fc['date'] = pd.to_datetime(best_fc['date'])
                fig.add_trace(go.Scatter(x=best_fc['date'], y=best_fc['prediction'],
                                         mode='lines+markers', name=f'Best Forecast ({results.get("best_model", "Unknown")})',
                                         line=dict(width=3)))
                # confidence band
                fig.add_trace(go.Scatter(
                    x=list(best_fc['date']) + list(best_fc['date'][::-1]),
                    y=list(best_fc['upper_bound']) + list(best_fc['lower_bound'][::-1]),
                    fill='toself', fillcolor='rgba(0,100,80,0.15)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip", showlegend=True, name='Confidence Interval'
                ))
            
            # plot ensemble if present
            if results.get('ensemble_forecast'):
                ens_fc = pd.DataFrame(results['ensemble_forecast'])
                ens_fc['date'] = pd.to_datetime(ens_fc['date'])
                fig.add_trace(go.Scatter(x=ens_fc['date'], y=ens_fc['prediction'],
                                         mode='lines', name='Ensemble Forecast', line=dict(dash='dot')))
            
            fig.update_layout(
                title='Forecast vs Historical',
                xaxis_title='Date',
                yaxis_title=value_col,
                template='plotly_white',
                height=450
            )
            results['forecast_plot'] = fig
        except Exception as e:
            results['forecast_plot'] = None
            results.setdefault('errors', []).append(f"Forecast plot error: {str(e)}")
        
        # attach prophet components into results['prophet_components'] (already set during prophet train)
        # attach ARIMA diagnostics already added during ARIMA training into results['arima_diagnostics']
        # results['prophet_components'] = AdvancedForecastAgent._format_prophet_components(results['prophet_components'])
        return results
