# 📁 Project Structure - AI Data Intelligence Agent v7.0

## 🗂️ Directory Tree

```
CapstoneAgents/
│
├── 📄 app.py                          # Original stable app
├── 📄 app_extended.py                 # ✨ NEW: 4-mode advanced UI
├── 📄 config.py                       # Configuration settings
├── 📄 demo.py                         # Demo script
├── 📄 start.py                        # Startup script
├── 📄 requirements.txt                # ✨ UPDATED: New dependencies
├── 📄 Dockerfile                      # Docker configuration
│
├── 📚 README.md                       # Original README
├── 📚 IMPLEMENTATION_GUIDE.md         # ✨ NEW: Complete implementation guide
├── 📚 QUICK_START.md                  # ✨ NEW: Quick start guide
├── 📚 AUDIT_SUMMARY.md                # ✨ NEW: Audit results summary
├── 📚 FEATURES_MATRIX.md              # ✨ NEW: Features comparison
├── 📚 PROJECT_STRUCTURE.md            # ✨ NEW: This file
│
├── .streamlit/
│   └── secrets.toml                   # API keys (Streamlit Cloud)
│
├── .env                               # Environment variables (local)
│
├── agents/                            # 🤖 Agent modules
│   ├── __init__.py                    # ✨ UPDATED: New exports
│   ├── preprocessor.py                # Data cleaning
│   ├── analyzer.py                    # KPI computation
│   ├── feature_engineer.py            # Feature engineering
│   ├── forecast_agent.py              # Basic forecasting
│   ├── anomaly_agent.py               # Anomaly detection
│   ├── llm_agent.py                   # LLM chat
│   │
│   ├── timeseries_processor.py        # ✨ NEW: Time-series toolkit
│   ├── advanced_forecast_agent.py     # ✨ NEW: ARIMA/Prophet/RF
│   ├── monetary_aggregates.py         # ✨ NEW: M1/M3/CPI analysis
│   ├── rag_agent.py                   # ✨ NEW: RAG system
│   └── enhanced_llm_agent.py          # ✨ NEW: RAG-enhanced LLM
│
└── utils/                             # 🛠️ Utility modules
    ├── __init__.py
    ├── data_loader.py                 # Universal data loader
    └── schema_detector.py             # Schema detection
```

---

## 📊 Component Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    STREAMLIT UI                         │
│  ┌──────────┬──────────┬──────────┬──────────┐         │
│  │ Single   │ Multi-   │ Monetary │ Document │         │
│  │ Dataset  │ CSV      │ Analysis │ RAG      │         │
│  └──────────┴──────────┴──────────┴──────────┘         │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  DATA PROCESSING LAYER                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ PreprocessorAgent → Clean & Transform Data       │  │
│  │ AnalyzerAgent → Compute KPIs                     │  │
│  │ TimeSeriesProcessor → Decompose & Detect         │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  ANALYSIS LAYER                         │
│  ┌────────────────┬────────────────┬─────────────────┐ │
│  │ AdvancedForecast│ MonetaryAnalysis│ RAG System    │ │
│  │ - ARIMA        │ - M1/M3         │ - PDF Load    │ │
│  │ - Prophet      │ - CPI           │ - Embeddings  │ │
│  │ - Random Forest│ - Correlation   │ - FAISS Store │ │
│  │ - Ensemble     │ - YOY/MOM       │ - Retrieval   │ │
│  └────────────────┴────────────────┴─────────────────┘ │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  AI INTELLIGENCE LAYER                  │
│  ┌──────────────────────────────────────────────────┐  │
│  │ EnhancedLLMAgent                                 │  │
│  │ ┌─────────────┐  ┌─────────────┐                │  │
│  │ │ Dataset     │  │ Document    │                │  │
│  │ │ Context     │  │ RAG Context │                │  │
│  │ └──────┬──────┘  └──────┬──────┘                │  │
│  │        └────────┬────────┘                       │  │
│  │                 ▼                                 │  │
│  │        ┌────────────────┐                        │  │
│  │        │  LLM Provider  │                        │  │
│  │        │ Gemini/Groq/   │                        │  │
│  │        │ GPT/Claude     │                        │  │
│  │        └────────────────┘                        │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow Diagram

### Single Dataset Analysis
```
User Upload CSV
    │
    ▼
PreprocessorAgent
    │ Clean Data
    ▼
TimeSeriesProcessor
    │ Detect Dates, Seasonality
    ▼
AdvancedForecastAgent
    │ Train ARIMA/Prophet/RF
    ▼
AnomalyAgent
    │ Detect Outliers
    ▼
AnalyzerAgent
    │ Compute KPIs
    ▼
EnhancedLLMAgent
    │ Generate Insights
    ▼
Display Results + Chat UI
```

### Multi-CSV Merge
```
User Upload Multiple CSVs
    │
    ▼
Merge on Common Column
    │
    ▼
[Same as Single Dataset Flow]
```

### Monetary Analysis
```
User Upload M1/M3/CPI Data
    │
    ▼
MonetaryAggregatesAnalyzer
    │ Compute YOY/MOM
    │ Calculate Correlations
    │ Analyze Inflation Impact
    ▼
Display Heatmaps + Insights
```

### Document RAG
```
User Upload PDF/DOCX
    │
    ▼
RAGAgent
    │ Parse Documents
    │ Chunk Text
    │ Generate Embeddings
    ▼
FAISS Vector Store
    │
    ▼
User Query → Retrieve Docs → LLM → Answer
```

---

## 🧩 Module Dependencies

```
app.py / app_extended.py
    │
    ├─→ agents/preprocessor.py
    │   └─→ pandas, numpy
    │
    ├─→ agents/analyzer.py
    │   └─→ pandas, numpy
    │
    ├─→ agents/timeseries_processor.py
    │   └─→ pandas, statsmodels
    │
    ├─→ agents/advanced_forecast_agent.py
    │   ├─→ statsmodels (ARIMA)
    │   ├─→ prophet (Prophet)
    │   └─→ sklearn (Random Forest)
    │
    ├─→ agents/monetary_aggregates.py
    │   └─→ pandas, numpy
    │
    ├─→ agents/rag_agent.py
    │   ├─→ sentence-transformers
    │   ├─→ faiss-cpu
    │   ├─→ PyPDF2
    │   └─→ python-docx
    │
    ├─→ agents/enhanced_llm_agent.py
    │   ├─→ agents/llm_agent.py
    │   ├─→ agents/rag_agent.py
    │   └─→ langchain
    │
    └─→ utils/
        ├─→ data_loader.py
        └─→ schema_detector.py
```

---

## 🎯 Entry Points

### For End Users
1. **app.py** - Stable, production-ready, original features
2. **app_extended.py** - Advanced, 4-mode interface, all new features

### For Developers
```python
# Import individual agents
from agents import (
    PreprocessorAgent,
    AdvancedForecastAgent,
    MonetaryAggregatesAnalyzer,
    FinancialRAGAgent,
    EnhancedLLMAgent
)

# Use programmatically
df = pd.read_csv('data.csv')
preprocessor = PreprocessorAgent(schema)
df_clean = preprocessor.preprocess(df)

# Run forecasting
forecast_results = AdvancedForecastAgent.train_all_models(
    df_clean, 'Date', 'Sales', horizon=14
)
```

---

## 📦 Package Management

### Core Dependencies (Always Required)
```
streamlit
pandas
numpy
plotly
scikit-learn
langchain (core)
```

### Optional Dependencies (Feature-Specific)
```
statsmodels         → ARIMA forecasting
prophet             → Prophet forecasting
sentence-transformers → RAG embeddings
faiss-cpu          → Vector search
PyPDF2             → PDF loading
python-docx        → DOCX loading
```

### Installation Strategy
```bash
# Minimal install (basic features only)
pip install streamlit pandas numpy plotly scikit-learn

# Full install (all features)
pip install -r requirements.txt

# Selective install (choose features)
pip install -r requirements.txt --no-deps
pip install streamlit pandas numpy plotly  # Core only
pip install prophet  # Add forecasting
pip install sentence-transformers faiss-cpu  # Add RAG
```

---

## 🔧 Configuration Files

### .env (Local Development)
```bash
GOOGLE_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
DEFAULT_LLM_PROVIDER=gemini
```

### .streamlit/secrets.toml (Streamlit Cloud)
```toml
[gemini]
api_key = "your_key"

[groq]
api_key = "your_key"

DEFAULT_LLM_PROVIDER = "gemini"
```

### config.py (Application Settings)
```python
# Model parameters
DEFAULT_MODEL = {
    'openai': 'gpt-4-turbo-preview',
    'anthropic': 'claude-3-5-sonnet',
    'groq': 'llama-3.3-70b',
    'gemini': 'gemini-2.0-flash'
}

# Analysis parameters
MAX_LAG_WINDOWS = [1, 7, 30, 90]
ROLLING_WINDOWS = [7, 30, 90]
```

---

## 📈 Scalability Notes

### Performance Limits
```
CSV Size:        < 500 MB recommended
Row Count:       < 1M rows optimal
ARIMA:           < 1,000 observations
Prophet:         Any size (handles millions)
RAG Documents:   < 100 PDFs recommended
Vector Store:    Scales to 100K+ chunks
```

### Optimization Tips
```
1. Use parquet for large files (faster than CSV)
2. Sample large datasets before analysis
3. Use Prophet instead of ARIMA for big data
4. Cache vector stores to disk
5. Use batch processing for multiple analyses
```

---

## 🚀 Deployment Options

### Local
```bash
streamlit run app_extended.py
```

### Docker
```bash
docker build -t ai-data-agent .
docker run -p 8501:8501 ai-data-agent
```

### Streamlit Cloud
```
1. Push to GitHub
2. Connect at share.streamlit.io
3. Add secrets in dashboard
4. Deploy
```

### Kubernetes
```yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-data-agent
spec:
  selector:
    app: ai-data-agent
  ports:
  - port: 8501
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-data-agent
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: app
        image: ai-data-agent:latest
        ports:
        - containerPort: 8501
```

---

## 🎯 Version History

| Version | Date | Major Changes |
|---------|------|---------------|
| v1.0 | Early 2024 | Initial release |
| v2.0 | Mar 2024 | Added forecasting |
| v3.0 | May 2024 | LLM integration |
| v4.0 | Aug 2024 | Multi-provider LLM |
| v5.0 | Oct 2024 | Anomaly detection |
| v6.0 | Nov 2024 | Bug fixes, stability |
| **v7.0** | **Nov 28, 2024** | **ARIMA/Prophet/RAG/Monetary** |

---

## 📞 Quick Reference

| Task | Command |
|------|---------|
| Install | `pip install -r requirements.txt` |
| Run Original | `streamlit run app.py` |
| Run Extended | `streamlit run app_extended.py` |
| Test Import | `python -c "from agents import *"` |
| Check Version | `streamlit --version` |
| Clear Cache | `streamlit cache clear` |

---

**Last Updated:** November 28, 2025  
**Maintained By:** AI Data Intelligence Team  
**License:** Open Source (check repository for details)
