"""
Universal Data Preprocessor - Works with any dataset type
Clean, robust, and defensive. Keeps logs and mapping.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
import re
warnings.filterwarnings("ignore")


class PreprocessorAgent:
    """Universal data preprocessing for any domain"""

    def __init__(self, schema: Dict = None):
        """
        Initialize with detected schema (optional).
        Args:
            schema: Schema dictionary from schema detector (optional)
        """
        self.schema = schema or {}
        self.processing_log: List[Dict] = []
        self.column_mapping: Dict[str, str] = {}  # Track original -> standardized column names

    # -------------------------
    # Public entry
    # -------------------------
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess dataset
        Args:
            df: Raw DataFrame
        Returns:
            Cleaned DataFrame
        """
        if df is None:
            raise ValueError("Input dataframe is None")

        df_clean = df.copy()
        initial_shape = df_clean.shape

        # 0. Defensive: drop entirely empty columns
        df_clean = self._drop_empty_columns(df_clean)

        # 1. Remove duplicates
        df_clean = self._remove_duplicates(df_clean)

        # 2. Standardize column names
        df_clean = self._standardize_columns(df_clean)

        # 3. Recompute schema keys to standardized names for internal use
        self._normalize_schema_keys()

        # 4. Handle datetime columns (detect if schema missing)
        df_clean = self._process_datetime(df_clean)

        # 5. Clean numeric columns
        df_clean = self._clean_numeric(df_clean)

        # 6. Clean categorical columns
        df_clean = self._clean_categorical(df_clean)

        # 7. Handle missing values
        df_clean = self._handle_missing(df_clean)

        # 8. Remove ID columns (not useful for modeling)
        df_clean = self._remove_id_columns(df_clean)

        # 9. Ensure index is reset and types sane
        df_clean = df_clean.reset_index(drop=True)

        final_shape = df_clean.shape
        self.processing_log.append({
            'step': 'complete',
            'initial_shape': initial_shape,
            'final_shape': final_shape,
            'rows_removed': initial_shape[0] - final_shape[0],
            'cols_removed': initial_shape[1] - final_shape[1],
            'column_mapping': self.column_mapping
        })

        return df_clean

    # -------------------------
    # Helpers
    # -------------------------
    def _drop_empty_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_cols = df.shape[1]
        non_empty = [c for c in df.columns if not (df[c].isnull().all())]
        if len(non_empty) < initial_cols:
            removed = list(set(df.columns) - set(non_empty))
            df = df[non_empty].copy()
            self.processing_log.append({
                'step': 'drop_empty_columns',
                'columns_removed': removed
            })
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        initial_len = len(df)
        df_clean = df.drop_duplicates().reset_index(drop=True)
        removed = initial_len - len(df_clean)

        if removed > 0:
            self.processing_log.append({
                'step': 'remove_duplicates',
                'rows_removed': removed,
                'percentage': removed / initial_len * 100
            })
        return df_clean

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and track mapping"""
        original_cols = df.columns.tolist()
        standardized_cols = []
        for c in original_cols:
            s = str(c).strip()
            # Replace spaces and special chars with underscore, collapse multiple underscores, uppercase
            s = re.sub(r'[^\w\s]', '', s)  # remove punctuation
            s = re.sub(r'\s+', '_', s)
            s = s.upper()
            s = re.sub(r'__+', '_', s)
            standardized_cols.append(s)

        # Track mapping for debugging
        self.column_mapping = dict(zip(original_cols, standardized_cols))
        df.columns = standardized_cols
        self.processing_log.append({
            'step': 'standardize_columns',
            'mapping': self.column_mapping
        })
        return df

    def _normalize_schema_keys(self):
        """If a schema was passed with original names, normalize them to standardized names"""
        if not self.schema:
            return

        def to_std(c):
            return str(c).strip().upper().replace(' ', '_')

        normalized = {}
        for k, v in self.schema.items():
            if isinstance(v, list):
                normalized[k] = [to_std(x) for x in v]
            else:
                normalized[k] = v
        self.schema = normalized

    def _process_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert datetime columns to proper dtype.
        If schema doesn't include datetime columns, try to auto-detect a date column.
        """
        datetime_cols = self.schema.get('datetime_columns', [])

        # Auto-detect if none were provided
        if not datetime_cols:
            detected = self._auto_detect_date_columns(df)
            if detected:
                datetime_cols = detected
                self.schema['datetime_columns'] = detected

        for col in datetime_cols:
            col_upper = str(col).upper().replace(' ', '_')
            if col_upper in df.columns:
                try:
                    df[col_upper] = pd.to_datetime(df[col_upper], errors='coerce')
                    self.processing_log.append({
                        'step': 'convert_datetime',
                        'column': col_upper
                    })
                except Exception as e:
                    self.processing_log.append({
                        'step': 'convert_datetime',
                        'column': col_upper,
                        'error': str(e)
                    })
        return df

    def _auto_detect_date_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Heuristic to find likely datetime columns. Returns list of standardized column names.
        """
        date_candidates = []
        DATE_KEYWORDS = ['DATE', 'TIME', 'TIMESTAMP', 'DATETIME', 'DAY', 'MONTH', 'YEAR']
        for col in df.columns:
            cu = str(col).upper()
            if any(k in cu for k in DATE_KEYWORDS):
                # quick parse test
                try:
                    sample = pd.to_datetime(df[col].dropna().head(50), errors='coerce')
                    if sample.notna().sum() > 0:
                        date_candidates.append(cu)
                except Exception:
                    continue

        # If none found by name, try parsing object columns
        if not date_candidates:
            for col in df.columns:
                if df[col].dtype == 'object' or np.issubdtype(df[col].dtype, np.integer):
                    try:
                        sample = pd.to_datetime(df[col].dropna().head(100), errors='coerce')
                        if len(sample) > 0 and sample.notna().sum() / len(sample) > 0.6:
                            date_candidates.append(str(col).upper())
                    except Exception:
                        continue

        return list(dict.fromkeys(date_candidates))  # dedupe, preserve order

    def _clean_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric columns based on hints and fallback numeric conversion"""
        numeric_hints = [
            "SALES", "REVENUE", "AMOUNT", "PRICE", "TOTAL", "QTY",
            "QUANTITY", "PROFIT", "COST", "VALUE", "SCORE", "RATE"
        ]

        # 1) For hinted columns, aggressively remove non-numeric chars
        for col in df.columns:
            if any(hint in col for hint in numeric_hints):
                try:
                    df[col] = pd.to_numeric(
                        df[col].astype(str).str.replace('[^0-9.-]', '', regex=True),
                        errors='coerce'
                    )
                    self.processing_log.append({'step': '_clean_numeric_hint', 'column': col})
                except Exception:
                    continue

        # 2) For columns that are numeric-like but stored as object, attempt conversion
        for col in df.columns:
            if df[col].dtype == 'object':
                # check if most values parse as numeric
                sample = df[col].dropna().astype(str).head(200)
                if len(sample) > 0:
                    parsed = pd.to_numeric(sample.str.replace('[^0-9.-]', '', regex=True), errors='coerce')
                    if parsed.notna().sum() / len(sample) > 0.8:
                        try:
                            df[col] = pd.to_numeric(df[col].astype(str).str.replace('[^0-9.-]', '', regex=True),
                                                   errors='coerce')
                            self.processing_log.append({'step': '_clean_numeric_guess', 'column': col})
                        except Exception:
                            pass

        return df

    def _clean_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean categorical columns: strip/upper and collapse long tails if necessary"""
        categorical_cols = self.schema.get('categorical_columns', [])

        # If schema not provided, infer small cardinality object columns as categorical
        if not categorical_cols:
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].nunique() / max(1, len(df)) < 0.5:
                    categorical_cols.append(col)

        # Standardize category strings
        for col in categorical_cols:
            col_upper = str(col).upper().replace(' ', '_')
            if col_upper in df.columns:
                try:
                    # strip + uppercase
                    df[col_upper] = df[col_upper].astype(str).str.strip()
                    # replace empty-like strings with UNKNOWN
                    df[col_upper] = df[col_upper].replace({'': np.nan, ' ': np.nan, 'NONE': np.nan})
                    df[col_upper] = df[col_upper].fillna('UNKNOWN')
                    # Keep length reasonable for text columns
                    df[col_upper] = df[col_upper].astype(str)
                    self.processing_log.append({'step': '_clean_categorical', 'column': col_upper})
                except Exception:
                    continue

        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values intelligently"""
        # Numeric columns: fill with 0 or median
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            n_missing = int(df[col].isnull().sum())
            if n_missing > 0:
                # Use 0 for likely counts/amounts, median for others
                col_upper = str(col).upper()
                fill_value = 0 if any(k in col_upper for k in ['QTY', 'QUANTITY', 'COUNT', 'AMOUNT']) else df[col].median()
                df[col] = df[col].fillna(fill_value)
                self.processing_log.append({'step': 'fill_numeric', 'column': col, 'fill_value': float(fill_value)})

        # Categorical/Object columns: fill with 'UNKNOWN'
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in object_cols:
            n_missing = int(df[col].isnull().sum())
            if n_missing > 0:
                df[col] = df[col].fillna('UNKNOWN')
                self.processing_log.append({'step': 'fill_object', 'column': col})

        # Datetime columns: forward fill then backward fill
        datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns.tolist()
        for col in datetime_cols:
            df[col] = df[col].ffill().bfill()
            self.processing_log.append({'step': 'fill_datetime', 'column': col})

        return df

    def _remove_id_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove ID columns that don't add value to analysis"""
        id_cols = self.schema.get('id_columns', [])
        cols_to_remove = []
        for col in id_cols:
            col_upper = str(col).upper().replace(' ', '_')
            if col_upper in df.columns:
                cols_to_remove.append(col_upper)

        if cols_to_remove:
            df = df.drop(columns=cols_to_remove)
            self.processing_log.append({
                'step': 'remove_id_columns',
                'columns_removed': cols_to_remove
            })

        return df

    # -------------------------
    # Introspection
    # -------------------------
    def get_processing_summary(self) -> Dict:
        """Get summary of preprocessing steps"""
        return {
            'steps_performed': len(self.processing_log),
            'log': self.processing_log,
            'column_mapping': self.column_mapping
        }
