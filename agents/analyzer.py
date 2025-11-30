"""
Universal Data Analyzer
✅ FIXED - Better error messages and validation
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

class AnalyzerAgent:
    """Compute universal KPIs and statistics"""
    
    @staticmethod
    def analyze(df: pd.DataFrame, target_col: str, schema: Dict) -> Dict:
        """
        Analyze dataset and compute KPIs
        """
        # Ensure uppercase
        target_col_upper = target_col.upper().replace(' ', '_') if target_col else None
        
        # Validate existence
        if target_col_upper and target_col_upper not in df.columns:
            raise ValueError(
                f"Target column '{target_col_upper}' not found in DataFrame.\n"
                f"Available columns: {list(df.columns)}\n\n"
                f"💡 Likely cause: The column was removed during preprocessing.\n"
                f"🔧 Solution: Update preprocessor.py to protect numeric columns."
            )
        
        # Validate numeric
        if target_col_upper and not pd.api.types.is_numeric_dtype(df[target_col_upper]):
            raise ValueError(
                f"Target column '{target_col_upper}' must be numeric.\n"
                f"Current type: {df[target_col_upper].dtype}"
            )
        
        kpis = {
            'dataset_info': {
                'rows': len(df),
                'cols': len(df.columns),
                'target_column': target_col_upper,
                'domain': schema.get('detected_domain', 'general'),
                'all_columns': list(df.columns)
            },
            'target_metrics': {},
            'top_categories': {}
        }
        
        # Target statistics
        if target_col_upper:
            try:
                vals = df[target_col_upper].dropna()
                
                if len(vals) == 0:
                    kpis['target_metrics'] = {'error': 'No valid data'}
                    return kpis
                
                kpis['target_metrics'] = {
                    'total': float(vals.sum()),
                    'mean': float(vals.mean()),
                    'median': float(vals.median()),
                    'std': float(vals.std()) if len(vals) > 1 else 0.0,
                    'min': float(vals.min()),
                    'max': float(vals.max()),
                    'q25': float(vals.quantile(0.25)),
                    'q75': float(vals.quantile(0.75)),
                    'non_null_count': int(vals.count())
                }
            except Exception as e:
                kpis['target_metrics'] = {'error': str(e)}
        else:
            kpis['target_metrics'] = {'error': 'No target column specified'}
        
        # Top categories
        categorical_cols = schema.get('categorical_columns', [])
        
        if target_col_upper:
            for cat_col in categorical_cols[:5]:
                col_upper = cat_col.upper().replace(' ', '_')
                
                if col_upper not in df.columns or col_upper == target_col_upper:
                    continue
                
                try:
                    top_items = (
                        df.groupby(col_upper)[target_col_upper]
                        .sum()
                        .sort_values(ascending=False)
                        .head(10)
                    )
                    kpis['top_categories'][col_upper] = top_items.to_dict()
                except:
                    pass
        
        # Time analysis
        datetime_cols = schema.get('datetime_columns', [])
        
        if target_col_upper and datetime_cols:
            date_col = datetime_cols[0].upper().replace(' ', '_')
            
            if date_col in df.columns:
                try:
                    df_sorted = df.sort_values(date_col)
                    date_min = df_sorted[date_col].min()
                    date_max = df_sorted[date_col].max()
                    
                    kpis['time_analysis'] = {
                        'date_range': {
                            'start': str(date_min),
                            'end': str(date_max),
                            'days': (date_max - date_min).days if pd.notna(date_min) and pd.notna(date_max) else 0
                        }
                    }
                    
                    df_sorted['MONTH'] = df_sorted[date_col].dt.to_period('M')
                    monthly = df_sorted.groupby('MONTH')[target_col_upper].sum()
                    kpis['time_analysis']['monthly_trend'] = {
                        str(k): float(v) for k, v in monthly.to_dict().items()
                    }
                except:
                    pass
        
        return kpis
