"""
Universal Data Loader - Supports CSV, Excel, JSON, Parquet
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

from config import FILE_ENCODINGS, LARGE_FILE_WARNING_MB


class UniversalDataLoader:
    """Load and validate data from multiple file formats"""
    
    @staticmethod
    def load_file(file_path: str, file_obj=None) -> Tuple[pd.DataFrame, dict]:
        """
        Load data from file or file object
        
        Args:
            file_path: Path to file or file extension
            file_obj: Streamlit UploadedFile object or file-like object
            
        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        metadata = {
            'file_name': file_path if isinstance(file_path, str) else 'uploaded_file',
            'file_size_mb': 0,
            'encoding': None,
            'load_method': None
        }
        
        try:
            # Determine file type
            if file_path.lower().endswith('.csv'):
                df, encoding = UniversalDataLoader._load_csv(file_obj if file_obj else file_path)
                metadata['encoding'] = encoding
                metadata['load_method'] = 'csv'
                
            elif file_path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_obj if file_obj else file_path, engine='openpyxl')
                metadata['load_method'] = 'excel'
                
            elif file_path.lower().endswith('.json'):
                df = pd.read_json(file_obj if file_obj else file_path)
                metadata['load_method'] = 'json'
                
            elif file_path.lower().endswith('.parquet'):
                df = pd.read_parquet(file_obj if file_obj else file_path)
                metadata['load_method'] = 'parquet'
                
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
            
            # Calculate file size if possible
            if file_obj and hasattr(file_obj, 'size'):
                metadata['file_size_mb'] = file_obj.size / (1024 * 1024)
            
            metadata['rows'] = len(df)
            metadata['columns'] = len(df.columns)
            
            return df, metadata
            
        except Exception as e:
            raise RuntimeError(f"Failed to load file: {str(e)}")
    
    @staticmethod
    def _load_csv(file_path) -> Tuple[pd.DataFrame, str]:
        """Load CSV with automatic encoding detection"""
        for encoding in FILE_ENCODINGS:
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    engine='python',
                    on_bad_lines='skip'
                )
                return df, encoding
            except:
                continue
        
        raise ValueError("Could not read CSV with any supported encoding")
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> dict:
        """
        Validate DataFrame and return quality metrics
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'info': {}
        }
        
        # Check minimum requirements
        if len(df) < 10:
            validation['errors'].append("Dataset too small (minimum 10 rows required)")
            validation['is_valid'] = False
        
        if len(df.columns) < 2:
            validation['errors'].append("Dataset must have at least 2 columns")
            validation['is_valid'] = False
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            validation['warnings'].append(f"Found {duplicates} duplicate rows ({duplicates/len(df)*100:.1f}%)")
        
        # Check for missing values
        missing_total = df.isnull().sum().sum()
        if missing_total > 0:
            missing_pct = missing_total / (len(df) * len(df.columns)) * 100
            validation['warnings'].append(f"Missing values: {missing_total} ({missing_pct:.1f}%)")
        
        # Check for completely empty columns
        empty_cols = [col for col in df.columns if df[col].isnull().all()]
        if empty_cols:
            validation['warnings'].append(f"Completely empty columns: {', '.join(empty_cols)}")
        
        # Data quality metrics
        validation['info'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'duplicate_rows': int(duplicates),
            'missing_values': int(missing_total),
            'missing_percentage': float(missing_pct) if missing_total > 0 else 0.0,
            'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024)
        }
        
        return validation
