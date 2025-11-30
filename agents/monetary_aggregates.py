"""
Monetary Aggregates Analysis Module
Supports analysis of:
- Money Supply (M1, M2, M3)
- CPI (Consumer Price Index)
- Repo Rate / Interest Rates
- Correlation analysis between monetary indicators
- YOY/MOM growth calculations
- Inflation impact analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


class MonetaryAggregatesAnalyzer:
    """Analyze monetary aggregates and macroeconomic indicators"""
    
    def __init__(self):
        self.data = {}
        self.correlations = {}
        self.trends = {}
    
    @staticmethod
    def load_monetary_dataset(file_path: str = None, df: pd.DataFrame = None,
                             date_col: str = 'Date', value_cols: List[str] = None) -> Dict:
        """
        Load and validate monetary aggregates dataset
        
        Args:
            file_path: Path to CSV/Excel file
            df: Pre-loaded DataFrame
            date_col: Name of date column
            value_cols: List of value column names (e.g., ['M1', 'M3', 'CPI', 'Repo_Rate'])
            
        Returns:
            Dict with loaded data and metadata
        """
        result = {
            'success': False,
            'data': None,
            'metadata': {}
        }
        
        try:
            # Load data
            if df is not None:
                data = df.copy()
            elif file_path:
                if file_path.endswith('.csv'):
                    data = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    data = pd.read_excel(file_path)
                else:
                    raise ValueError("Unsupported file format")
            else:
                raise ValueError("Either file_path or df must be provided")
            
            # Convert date column
            data[date_col] = pd.to_datetime(data[date_col])
            data = data.sort_values(date_col).reset_index(drop=True)
            
            # Auto-detect value columns if not provided
            if value_cols is None:
                monetary_keywords = ['m1', 'm2', 'm3', 'cpi', 'inflation', 'repo', 'rate', 'gdp']
                value_cols = []
                for col in data.columns:
                    if col != date_col and any(kw in col.lower() for kw in monetary_keywords):
                        value_cols.append(col)
            
            result.update({
                'success': True,
                'data': data,
                'metadata': {
                    'date_column': date_col,
                    'value_columns': value_cols,
                    'start_date': str(data[date_col].min().date()),
                    'end_date': str(data[date_col].max().date()),
                    'total_records': len(data),
                    'frequency': MonetaryAggregatesAnalyzer._detect_frequency(data[date_col])
                }
            })
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    @staticmethod
    def _detect_frequency(date_series: pd.Series) -> str:
        """Detect data frequency (daily, weekly, monthly, quarterly, yearly)"""
        if len(date_series) < 2:
            return 'unknown'
        
        deltas = date_series.diff().dropna()
        median_delta = deltas.median().days
        
        if median_delta <= 1:
            return 'daily'
        elif median_delta <= 7:
            return 'weekly'
        elif median_delta <= 31:
            return 'monthly'
        elif median_delta <= 92:
            return 'quarterly'
        else:
            return 'yearly'
    
    @staticmethod
    def calculate_yoy_mom_growth(df: pd.DataFrame, date_col: str, value_col: str) -> Dict:
        """
        Calculate Year-over-Year and Month-over-Month growth rates
        
        Returns:
            Dict with YOY and MOM growth data
        """
        result = {
            'yoy': [],
            'mom': [],
            'avg_yoy': 0.0,
            'avg_mom': 0.0,
            'latest_yoy': 0.0,
            'latest_mom': 0.0
        }
        
        try:
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            df_copy = df_copy.sort_values(date_col)
            
            # Month-over-Month
            df_copy['year_month'] = df_copy[date_col].dt.to_period('M')
            monthly = df_copy.groupby('year_month')[value_col].mean().reset_index()
            monthly['mom_growth'] = monthly[value_col].pct_change() * 100
            
            result['mom'] = monthly[['year_month', 'mom_growth']].dropna().to_dict('records')
            result['avg_mom'] = float(monthly['mom_growth'].mean()) if not monthly['mom_growth'].isna().all() else 0.0
            result['latest_mom'] = float(monthly['mom_growth'].iloc[-1]) if len(monthly) > 0 and not pd.isna(monthly['mom_growth'].iloc[-1]) else 0.0
            
            # Year-over-Year
            df_copy['year'] = df_copy[date_col].dt.year
            yearly = df_copy.groupby('year')[value_col].mean().reset_index()
            yearly['yoy_growth'] = yearly[value_col].pct_change() * 100
            
            result['yoy'] = yearly[['year', 'yoy_growth']].dropna().to_dict('records')
            result['avg_yoy'] = float(yearly['yoy_growth'].mean()) if not yearly['yoy_growth'].isna().all() else 0.0
            result['latest_yoy'] = float(yearly['yoy_growth'].iloc[-1]) if len(yearly) > 0 and not pd.isna(yearly['yoy_growth'].iloc[-1]) else 0.0
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    @staticmethod
    def analyze_correlation(df: pd.DataFrame, date_col: str, variables: List[str]) -> Dict:
        """
        Analyze correlations between monetary variables
        
        Args:
            df: DataFrame with monetary data
            date_col: Date column name
            variables: List of variable names to correlate
            
        Returns:
            Dict with correlation matrix and insights
        """
        result = {
            'correlation_matrix': {},
            'top_correlations': [],
            'insights': []
        }
        
        try:
            # Select only numeric columns
            numeric_data = df[variables].select_dtypes(include=[np.number])
            
            # Calculate correlation matrix
            corr_matrix = numeric_data.corr()
            result['correlation_matrix'] = corr_matrix.to_dict()
            
            # Extract top correlations (excluding diagonal)
            correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    var1 = corr_matrix.columns[i]
                    var2 = corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    
                    correlations.append({
                        'variable_1': var1,
                        'variable_2': var2,
                        'correlation': float(corr_val),
                        'strength': MonetaryAggregatesAnalyzer._correlation_strength(corr_val)
                    })
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            result['top_correlations'] = correlations[:10]
            
            # Generate insights
            for corr in correlations[:5]:
                if abs(corr['correlation']) > 0.7:
                    direction = 'positive' if corr['correlation'] > 0 else 'negative'
                    result['insights'].append(
                        f"Strong {direction} correlation ({corr['correlation']:.2f}) between "
                        f"{corr['variable_1']} and {corr['variable_2']}"
                    )
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    @staticmethod
    def _correlation_strength(corr_value: float) -> str:
        """Classify correlation strength"""
        abs_corr = abs(corr_value)
        if abs_corr >= 0.7:
            return 'Strong'
        elif abs_corr >= 0.4:
            return 'Moderate'
        elif abs_corr >= 0.2:
            return 'Weak'
        else:
            return 'Very Weak'
    
    @staticmethod
    def analyze_inflation_impact(df: pd.DataFrame, date_col: str, 
                                m3_col: str, cpi_col: str) -> Dict:
        """
        Analyze relationship between money supply (M3) and inflation (CPI)
        
        Returns:
            Dict with analysis results
        """
        result = {
            'correlation': 0.0,
            'lag_analysis': [],
            'insights': []
        }
        
        try:
            df_copy = df[[date_col, m3_col, cpi_col]].copy()
            df_copy = df_copy.dropna()
            
            # Direct correlation
            direct_corr = df_copy[m3_col].corr(df_copy[cpi_col])
            result['correlation'] = float(direct_corr)
            
            # Lag correlation analysis (check if M3 changes lead CPI changes)
            max_lag = min(12, len(df_copy) // 4)  # Up to 12 periods or 1/4 of data
            
            for lag in range(0, max_lag + 1):
                if lag == 0:
                    corr = direct_corr
                else:
                    m3_series = df_copy[m3_col].iloc[:-lag]
                    cpi_series = df_copy[cpi_col].iloc[lag:]
                    
                    if len(m3_series) > 0 and len(cpi_series) > 0:
                        corr = m3_series.corr(cpi_series)
                    else:
                        corr = 0
                
                result['lag_analysis'].append({
                    'lag_periods': lag,
                    'correlation': float(corr)
                })
            
            # Find optimal lag
            optimal_lag = max(result['lag_analysis'], key=lambda x: abs(x['correlation']))
            
            if optimal_lag['lag_periods'] > 0:
                result['insights'].append(
                    f"M3 changes show strongest correlation with CPI at {optimal_lag['lag_periods']} period(s) lag "
                    f"(correlation: {optimal_lag['correlation']:.3f})"
                )
            
            if abs(direct_corr) > 0.5:
                direction = 'positive' if direct_corr > 0 else 'negative'
                result['insights'].append(
                    f"Strong {direction} relationship detected between money supply (M3) and inflation (CPI)"
                )
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    @staticmethod
    def create_monetary_features(df: pd.DataFrame, date_col: str, 
                                 value_cols: List[str]) -> pd.DataFrame:
        """
        Create engineered features for monetary data analysis
        
        Features include:
        - Growth rates (YOY, MOM)
        - Moving averages
        - Volatility measures
        - Momentum indicators
        
        Returns:
            DataFrame with additional features
        """
        df_enhanced = df.copy()
        
        try:
            df_enhanced[date_col] = pd.to_datetime(df_enhanced[date_col])
            df_enhanced = df_enhanced.sort_values(date_col)
            
            for col in value_cols:
                if col not in df_enhanced.columns:
                    continue
                
                # Growth rates
                df_enhanced[f'{col}_MOM'] = df_enhanced[col].pct_change() * 100
                df_enhanced[f'{col}_YOY'] = df_enhanced[col].pct_change(12) * 100  # Assuming monthly data
                
                # Moving averages
                df_enhanced[f'{col}_MA3'] = df_enhanced[col].rolling(window=3, min_periods=1).mean()
                df_enhanced[f'{col}_MA6'] = df_enhanced[col].rolling(window=6, min_periods=1).mean()
                df_enhanced[f'{col}_MA12'] = df_enhanced[col].rolling(window=12, min_periods=1).mean()
                
                # Volatility (rolling std)
                df_enhanced[f'{col}_Volatility'] = df_enhanced[col].rolling(window=6, min_periods=1).std()
                
                # Momentum (current vs MA)
                df_enhanced[f'{col}_Momentum'] = ((df_enhanced[col] - df_enhanced[f'{col}_MA6']) / 
                                                  df_enhanced[f'{col}_MA6']) * 100
                
                # Rate of change
                df_enhanced[f'{col}_ROC_3M'] = ((df_enhanced[col] - df_enhanced[col].shift(3)) / 
                                                df_enhanced[col].shift(3)) * 100
            
            # Fill NaN values
            df_enhanced = df_enhanced.fillna(method='bfill').fillna(0)
            
        except Exception as e:
            print(f"Feature engineering error: {e}")
        
        return df_enhanced
    
    @staticmethod
    def generate_monetary_report(df: pd.DataFrame, date_col: str, 
                                 value_cols: List[str]) -> Dict:
        """
        Generate comprehensive monetary analysis report
        
        Returns:
            Dict with complete analysis
        """
        report = {
            'summary_statistics': {},
            'growth_analysis': {},
            'correlation_analysis': {},
            'trend_analysis': {},
            'insights': []
        }
        
        try:
            # Summary statistics for each variable
            for col in value_cols:
                if col in df.columns:
                    report['summary_statistics'][col] = {
                        'current': float(df[col].iloc[-1]) if len(df) > 0 else 0,
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'median': float(df[col].median())
                    }
                    
                    # Growth analysis
                    growth = MonetaryAggregatesAnalyzer.calculate_yoy_mom_growth(df, date_col, col)
                    report['growth_analysis'][col] = {
                        'latest_yoy': growth['latest_yoy'],
                        'latest_mom': growth['latest_mom'],
                        'avg_yoy': growth['avg_yoy'],
                        'avg_mom': growth['avg_mom']
                    }
            
            # Correlation analysis
            report['correlation_analysis'] = MonetaryAggregatesAnalyzer.analyze_correlation(
                df, date_col, value_cols
            )
            
            # Generate insights
            for col in value_cols:
                if col in report['growth_analysis']:
                    yoy = report['growth_analysis'][col]['latest_yoy']
                    if abs(yoy) > 10:
                        direction = 'increased' if yoy > 0 else 'decreased'
                        report['insights'].append(
                            f"{col} has {direction} by {abs(yoy):.1f}% year-over-year"
                        )
            
        except Exception as e:
            report['error'] = str(e)
        
        return report
