"""
Smart Schema Detection - Auto-detect column types and target variables
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re


class SchemaDetector:
    """Automatically detect column types and dataset characteristics"""
    
    @staticmethod
    def detect_schema(df: pd.DataFrame) -> Dict:
        """
        Analyze DataFrame and detect column types
        
        Returns:
            Dictionary with schema information
        """
        schema = {
            'datetime_columns': [],
            'numeric_columns': [],
            'categorical_columns': [],
            'text_columns': [],
            'id_columns': [],
            'target_candidates': [],
            'column_analysis': {}
        }
        
        for col in df.columns:
            col_info = SchemaDetector._analyze_column(df[col], col)
            schema['column_analysis'][col] = col_info
            
            # Classify column
            if col_info['type'] == 'datetime':
                schema['datetime_columns'].append(col)
            elif col_info['type'] == 'numeric':
                schema['numeric_columns'].append(col)
                # Check if potential target
                if col_info['is_target_candidate']:
                    schema['target_candidates'].append({
                        'column': col,
                        'score': col_info['target_score'],
                        'reason': col_info['target_reason']
                    })
            elif col_info['type'] == 'categorical':
                schema['categorical_columns'].append(col)
            elif col_info['type'] == 'id':
                schema['id_columns'].append(col)
            else:
                schema['text_columns'].append(col)
        
        # Sort target candidates by score
        schema['target_candidates'] = sorted(
            schema['target_candidates'],
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Detect dataset domain
        schema['detected_domain'] = SchemaDetector._detect_domain(df, schema)
        
        return schema
    
    @staticmethod
    def _analyze_column(series: pd.Series, col_name: str) -> Dict:
        """Analyze a single column"""
        col_info = {
            'type': 'unknown',
            'unique_count': series.nunique(),
            'null_count': series.isnull().sum(),
            'null_percentage': series.isnull().sum() / len(series) * 100,
            'is_target_candidate': False,
            'target_score': 0,
            'target_reason': ''
        }
        
        # Skip if all null
        if series.isnull().all():
            col_info['type'] = 'empty'
            return col_info
        
        # Check for ID column
        if SchemaDetector._is_id_column(series, col_name):
            col_info['type'] = 'id'
            return col_info
        
        # Check for datetime
        if SchemaDetector._is_datetime(series):
            col_info['type'] = 'datetime'
            col_info['date_range'] = {
                'min': str(series.min()),
                'max': str(series.max())
            }
            return col_info
        
        # Check for numeric
        if pd.api.types.is_numeric_dtype(series):
            col_info['type'] = 'numeric'
            col_info['stats'] = {
                'min': float(series.min()),
                'max': float(series.max()),
                'mean': float(series.mean()),
                'median': float(series.median()),
                'std': float(series.std())
            }
            
            # Check if target candidate
            target_check = SchemaDetector._check_target_candidate(series, col_name)
            col_info.update(target_check)
            
        # Check for categorical
        elif col_info['unique_count'] / len(series) < 0.5:  # Less than 50% unique
            col_info['type'] = 'categorical'
            col_info['categories'] = series.value_counts().head(10).to_dict()
            
        # Otherwise text
        else:
            col_info['type'] = 'text'
            col_info['avg_length'] = series.astype(str).str.len().mean()
        
        return col_info
    
    @staticmethod
    def _is_id_column(series: pd.Series, col_name: str) -> bool:
        """Check if column is likely an ID field"""
        # Check name patterns
        id_patterns = ['id', 'key', 'index', 'identifier', 'uuid', 'guid']
        if any(pattern in col_name.lower() for pattern in id_patterns):
            return True
        
        # Check if unique values = row count
        if series.nunique() == len(series):
            return True
        
        return False
    
    @staticmethod
    def _is_datetime(series: pd.Series) -> bool:
        """Check if column contains datetime values"""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        # Try parsing as datetime
        if series.dtype == 'object':
            try:
                pd.to_datetime(series.dropna().head(100), errors='raise')
                return True
            except:
                pass
        
        return False
    
    @staticmethod
    def _check_target_candidate(series: pd.Series, col_name: str) -> Dict:
        """Check if numeric column is a good target variable candidate"""
        result = {
            'is_target_candidate': False,
            'target_score': 0,
            'target_reason': ''
        }
        
        score = 0
        reasons = []
        
        # Check column name for target indicators
        target_keywords = [
            'sales', 'revenue', 'amount', 'total', 'value', 'price',
            'quantity', 'qty', 'profit', 'cost', 'spend', 'volume',
            'count', 'rate', 'score', 'target', 'label', 'outcome'
        ]
        
        col_lower = col_name.lower()
        for keyword in target_keywords:
            if keyword in col_lower:
                score += 30
                reasons.append(f"Column name contains '{keyword}'")
                break
        
        # Check if values are mostly positive (common for business metrics)
        positive_ratio = (series > 0).sum() / len(series)
        if positive_ratio > 0.8:
            score += 20
            reasons.append("Mostly positive values")
        
        # Check for reasonable variance
        if series.std() > 0:
            cv = series.std() / series.mean() if series.mean() != 0 else 0
            if 0.1 < cv < 10:  # Coefficient of variation in reasonable range
                score += 15
                reasons.append("Good variance")
        
        # Check data type (float is more likely target than int for some cases)
        if series.dtype == 'float64':
            score += 5
        
        # Penalize if too few unique values (might be categorical)
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.01:
            score -= 20
            reasons.append("Too few unique values")
        
        if score >= 25:  # Threshold for being a candidate
            result['is_target_candidate'] = True
            result['target_score'] = score
            result['target_reason'] = '; '.join(reasons)
        
        return result
    
    @staticmethod
    def _detect_domain(df: pd.DataFrame, schema: Dict) -> str:
        """Detect likely domain/industry of the dataset"""
        columns_lower = [col.lower() for col in df.columns]
        
        # Retail/E-commerce
        retail_keywords = ['sales', 'revenue', 'product', 'customer', 'order', 'quantity', 'price']
        if sum(any(k in col for k in retail_keywords) for col in columns_lower) >= 3:
            return 'retail_ecommerce'
        
        # Finance
        finance_keywords = ['account', 'transaction', 'balance', 'payment', 'credit', 'debit', 'amount']
        if sum(any(k in col for k in finance_keywords) for col in columns_lower) >= 3:
            return 'finance'
        
        # Healthcare
        health_keywords = ['patient', 'diagnosis', 'treatment', 'hospital', 'doctor', 'medical']
        if sum(any(k in col for k in health_keywords) for col in columns_lower) >= 2:
            return 'healthcare'
        
        # HR
        hr_keywords = ['employee', 'salary', 'department', 'hire', 'performance', 'tenure']
        if sum(any(k in col for k in hr_keywords) for col in columns_lower) >= 3:
            return 'human_resources'
        
        # IoT/Sensor
        iot_keywords = ['sensor', 'temperature', 'humidity', 'device', 'reading', 'measurement']
        if sum(any(k in col for k in iot_keywords) for col in columns_lower) >= 2:
            return 'iot_sensor'
        
        # Marketing
        marketing_keywords = ['campaign', 'impression', 'click', 'conversion', 'engagement', 'roi']
        if sum(any(k in col for k in marketing_keywords) for col in columns_lower) >= 2:
            return 'marketing'
        
        return 'general'
    
    @staticmethod
    def suggest_target_variable(schema: Dict, user_preference: Optional[str] = None) -> Tuple[str, str]:
        """
        Suggest the best target variable
        
        Args:
            schema: Schema dictionary from detect_schema
            user_preference: Optional user-specified target column
            
        Returns:
            Tuple of (target_column, reason)
        """
        if user_preference and user_preference in schema['column_analysis']:
            return user_preference, "User specified"
        
        if schema['target_candidates']:
            best = schema['target_candidates'][0]
            return best['column'], best['reason']
        
        # Fallback: use first numeric column
        if schema['numeric_columns']:
            return schema['numeric_columns'][0], "First numeric column (fallback)"
        
        raise ValueError("No suitable target variable found")
