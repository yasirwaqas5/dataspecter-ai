"""
========================================
AI Data Intelligence Agent v7.0 + KAGGLE 120/120
========================================
"""
import os


# ================== KAGGLE COMPETITION MODE ==================
# 🏆 Toggle ALL new features for 120/120 scoring
KAGGLE_MODE = True  # Flip to False = original app unchanged

USE_MULTI_AGENT = KAGGLE_MODE
USE_TRACING = KAGGLE_MODE
USE_EVAL = KAGGLE_MODE

# ================== STREAMLIT CONFIG ==================
PAGE_TITLE = "AI Data Intelligence Agent v7.0 - Kaggle Edition"
PAGE_ICON = "🏆"
LAYOUT = "wide"
SIDEBAR_STATE = "expanded"

# ================== LLM CONFIGURATION ==================
DEFAULT_MODEL = {
    'openai': 'gpt-4-turbo-preview',
    'anthropic': 'claude-3-5-sonnet',
    'groq': 'llama-3.3-70b',
    'gemini': 'gemini-2.0-flash'
}

GEMINI_MODELS = ['gemini-2.0-flash-exp', 'gemini-1.5-pro']
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

LLM_TEMPERATURE = 0.1  # Low temperature for analytical tasks
LLM_MAX_TOKENS = 4096

# ================== DATA PROCESSING ==================
MAX_FILE_SIZE_MB = 500  # Maximum upload file size
LARGE_FILE_WARNING_MB = 100  # Show warning for files larger than this
MAX_ROWS_PREVIEW = 1000  # Maximum rows to show in preview
SAMPLE_SIZE_LARGE_DATASETS = 100000  # Sample size for datasets > this

# ================== FEATURE ENGINEERING ==================
MAX_LAG_WINDOWS = [1, 2, 3, 7, 14, 21, 28, 30, 60, 90]
ROLLING_WINDOWS = [3, 7, 14, 30, 90]
EWM_SPANS = [7, 14, 30]
FOURIER_TERMS = 5

# ================== MODEL TRAINING ==================
VALIDATION_SPLIT_DAYS = 14  # Days to use for validation
CV_SPLITS = 5  # Cross-validation splits
OPTUNA_TRIALS = 20  # Hyperparameter optimization trials

# ================== ANOMALY DETECTION ==================
ANOMALY_CONTAMINATION = 0.02  # Expected proportion of anomalies
Z_SCORE_THRESHOLD = 3.0
LOF_NEIGHBORS = 20

# ================== VISUALIZATION ==================
PLOT_HEIGHT = 600
PLOT_WIDTH = 1200
COLOR_SCHEME = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800',
    'info': '#17a2b8'
}

# ================== RAG/FORECAST ==================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
FORECAST_HORIZON = 14
MIN_DATA_POINTS = 40

# ================== SUPPORTED FILE TYPES ==================
SUPPORTED_FILE_TYPES = ["csv", "xlsx", "xls", "json", "parquet"]
FILE_ENCODINGS = ["utf-8", "latin1", "ISO-8859-1", "cp1252"]

# ================== AGENT SYSTEM PROMPT ==================
AGENT_SYSTEM_PROMPT = """You are an expert AI Data Analyst with deep knowledge in:
- Statistical analysis and data science
- Time series forecasting
- Anomaly detection
- Machine learning model interpretation
- Business intelligence and KPI analysis

Your role is to help users understand their data through:
1. Answering analytical questions with precise, data-backed responses
2. Generating insights from statistical analysis
3. Explaining model predictions and feature importance
4. Providing actionable business recommendations
5. Identifying trends, patterns, and anomalies

Always:
- Be concise and precise
- Use actual numbers from the data
- Explain technical concepts in business terms when needed
- Provide actionable insights
- Cite specific metrics and visualizations when available

Available tools:
- get_kpis(): Retrieve key performance indicators
- get_forecast(): Get forecasting results
- get_anomalies(): Fetch detected anomalies
- get_model_performance(): Get model metrics
- get_feature_importance(): Get top influential features
- get_data_summary(): Get dataset statistics
- query_data(question): Perform custom data queries
"""