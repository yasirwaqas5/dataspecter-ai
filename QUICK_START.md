# 🚀 Quick Start Guide - AI Data Intelligence Agent v7.0

## ⚡ 5-Minute Setup

### 1. Install Dependencies
```bash
cd CapstoneAgents
pip install -r requirements.txt
```

### 2. Set API Key (Choose One)

**Option A: Quick Test (Manual Input)**
- Just run the app and enter key in sidebar

**Option B: Environment Variable**
```bash
# Windows PowerShell
$env:GOOGLE_API_KEY="your_key_here"

# Linux/Mac
export GOOGLE_API_KEY="your_key_here"
```

### 3. Run the App
```bash
streamlit run app.py
```

---

## 🎯 Common Use Cases

### Use Case 1: Basic Sales Analysis
**Goal:** Analyze sales data and get forecasts

```python
# In Streamlit app:
1. Upload: sales_data.csv
2. Select Target: "Sales" or "Revenue"
3. Select Date: "Date" column
4. Click "Run Complete Analysis"

# Results:
- KPIs (total, average, median)
- 14-day forecast
- Anomaly detection
- AI chat for questions
```

---

### Use Case 2: Monthly Financial Report
**Goal:** Analyze M1/M3/CPI data

**Step by Step:**

1. **Run Extended App:**
```bash
streamlit run app_extended.py
```

2. **Select "Monetary Analysis" mode**

3. **Upload CSV with columns:**
```
Date, M1, M3, CPI, Repo_Rate
2023-01-01, 1000, 5000, 120, 6.5
2023-02-01, 1050, 5100, 121, 6.5
...
```

4. **Select:**
- Date Column: `Date`
- Value Columns: `M1`, `M3`, `CPI`, `Repo_Rate`

5. **Click "Analyze Monetary Data"**

**You Get:**
- YOY/MOM growth for each variable
- Correlation heatmap
- Inflation impact analysis
- Key insights

---

### Use Case 3: Advanced Forecasting (ARIMA + Prophet)
**Goal:** Compare multiple forecasting models

**Python Code Example:**
```python
import pandas as pd
from agents.timeseries_processor import TimeSeriesProcessor
from agents.advanced_forecast_agent import AdvancedForecastAgent

# Load data
df = pd.read_csv('sales_data.csv')

# Prepare time-series
ts_processor = TimeSeriesProcessor()
ts_df = ts_processor.prepare_timeseries(
    df, 
    date_col='Date', 
    value_col='Sales', 
    freq='D'  # Daily frequency
)

# Train all models
results = AdvancedForecastAgent.train_all_models(
    ts_df,
    date_col='Date',
    value_col='Sales',
    horizon=14,
    enable_prophet=True,
    enable_arima=True
)

# Check results
print(f"Best Model: {results['best_model']}")
print(f"Models Trained: {list(results['models'].keys())}")

# View forecast
for day in results['best_forecast'][:3]:
    print(f"{day['date']}: ${day['prediction']:,.2f}")

# Model comparison
for model_perf in results['performance_comparison']:
    print(f"{model_perf['model']}: RMSE={model_perf['rmse']:.2f}")
```

**Output:**
```
Best Model: ARIMA
Models Trained: ['Random Forest', 'ARIMA', 'SARIMA', 'Prophet']

2024-01-01: $1,234.56
2024-01-02: $1,245.67
2024-01-03: $1,256.78

Random Forest: RMSE=45.23
ARIMA: RMSE=38.45
SARIMA: RMSE=40.12
Prophet: RMSE=42.89
```

---

### Use Case 4: Time-Series Decomposition
**Goal:** Understand seasonality and trends

**Python Code:**
```python
from agents.timeseries_processor import TimeSeriesProcessor

processor = TimeSeriesProcessor()

# Detect seasonality
seasonality = processor.detect_seasonality(df['Sales'])
print(f"Has Seasonality: {seasonality['has_seasonality']}")
print(f"Period: {seasonality['period']} days")

# Decompose series
decomp = processor.decompose_timeseries(
    df,
    date_col='Date',
    value_col='Sales',
    period=7  # Weekly pattern
)

if decomp['success']:
    trend_df = decomp['trend']
    seasonal_df = decomp['seasonal']
    
    print(f"Seasonal Strength: {decomp['seasonal_strength']:.2f}")
    
    # Plot in Streamlit
    import plotly.express as px
    fig = px.line(trend_df, x='Date', y='trend', title='Trend Component')
    st.plotly_chart(fig)
```

---

### Use Case 5: Document Q&A with RAG
**Goal:** Ask questions about financial PDFs

**Streamlit UI:**
1. Run: `streamlit run app_extended.py`
2. Select "Document RAG" mode
3. Upload: `annual_report_2023.pdf`
4. Click "Process Documents"
5. Ask: "What was the revenue in Q4?"

**Python Code:**
```python
from agents.rag_agent import FinancialRAGAgent

# Initialize
rag = FinancialRAGAgent()

# Load documents
result = rag.load_documents(file_paths=['report.pdf', 'analysis.docx'])
print(f"Loaded {result['total_chunks']} chunks")

# Create vector store
rag.create_vector_store()

# Search
results = rag.retrieve("What is the profit margin?", top_k=3)

for i, result in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(f"Source: {result['metadata']['source']}")
    print(f"Content: {result['content'][:200]}...")
```

---

### Use Case 6: Multi-CSV Merge
**Goal:** Combine regional sales data

**Files:**
- `sales_north.csv` (Date, Sales)
- `sales_south.csv` (Date, Sales)
- `sales_east.csv` (Date, Sales)

**Steps:**
1. Run: `streamlit run app_extended.py`
2. Select "Multi-CSV Merge"
3. Upload all 3 files
4. Merge Column: `Date`
5. Merge Type: `outer`
6. Click "Merge & Analyze"

**Result:**
```
Date       | Sales_north | Sales_south | Sales_east | Total
2024-01-01 | 1000       | 800         | 900        | 2700
2024-01-02 | 1100       | 850         | 950        | 2900
```

---

## 🔧 Code Snippets Library

### Snippet 1: Calculate YOY Growth
```python
from agents.timeseries_processor import TimeSeriesProcessor

growth = TimeSeriesProcessor.calculate_growth_metrics(
    df, 
    date_col='Date', 
    value_col='Revenue'
)

print(f"Average YOY Growth: {growth['avg_yoy_growth']:.2f}%")
print(f"Average MOM Growth: {growth['avg_mom_growth']:.2f}%")

# Plot YOY trend
import plotly.express as px
yoy_df = pd.DataFrame(growth['yoy_growth'])
fig = px.line(yoy_df, x='year', y='yoy_growth', title='YOY Growth Trend')
```

### Snippet 2: Ensemble Forecast
```python
from agents.advanced_forecast_agent import AdvancedForecastAgent

# Train multiple models
results = AdvancedForecastAgent.train_all_models(df, 'Date', 'Sales', horizon=14)

# Get ensemble forecast (weighted average)
ensemble = results['ensemble_forecast']

# Custom weights (favor ARIMA)
forecasts = {name: forecast for name, (_, forecast) in results['models'].items()}
custom_ensemble = AdvancedForecastAgent.ensemble_forecast(
    forecasts,
    weights={'ARIMA': 0.5, 'Prophet': 0.3, 'Random Forest': 0.2}
)
```

### Snippet 3: Correlation Analysis
```python
from agents.monetary_aggregates import MonetaryAggregatesAnalyzer

analyzer = MonetaryAggregatesAnalyzer()

# Check M3 vs CPI correlation
inflation_impact = analyzer.analyze_inflation_impact(
    df,
    date_col='Date',
    m3_col='M3',
    cpi_col='CPI'
)

print(f"Correlation: {inflation_impact['correlation']:.3f}")
print(f"Optimal Lag: {inflation_impact['lag_analysis'][0]['lag_periods']} periods")
```

### Snippet 4: Enhanced LLM with RAG
```python
from agents.enhanced_llm_agent import EnhancedLLMAgent
from agents.rag_agent import FinancialRAGAgent

# Setup RAG
rag = FinancialRAGAgent()
rag.load_documents(file_paths=['report.pdf'])
rag.create_vector_store()

# Create enhanced LLM
llm = EnhancedLLMAgent(
    analysis_context={'kpis': kpis, 'forecast': forecast},
    provider='gemini',
    api_key='your_key',
    rag_agent=rag
)

# Ask with RAG
answer = llm.ask("What drove the revenue increase?", retrieve_docs=True)

# Generate executive summary
summary = llm.generate_executive_summary()
print(summary)
```

---

## 📊 Sample Data Formats

### Sales Data (CSV)
```csv
Date,Product,Category,Sales,Quantity
2024-01-01,Widget A,Electronics,1234.56,10
2024-01-02,Widget B,Home,987.65,8
```

### Monetary Data (CSV)
```csv
Date,M1,M3,CPI,Repo_Rate
2024-01-01,50000,200000,120.5,6.5
2024-02-01,51000,202000,121.0,6.5
```

### Time-Series Data (CSV)
```csv
Date,Value
2024-01-01,100
2024-01-02,105
2024-01-03,102
```

---

## ⚙️ Configuration Tips

### For Best Forecasting Results:
- **Minimum data:** 40+ days for ARIMA, 60+ for SARIMA
- **Frequency:** Daily or weekly data works best
- **Stationarity:** ARIMA works best with stationary data
- **Seasonality:** Use SARIMA or Prophet for seasonal patterns

### For RAG Performance:
- **Chunk size:** 500-1000 characters optimal
- **Documents:** 5-50 PDFs recommended
- **Embedding model:** sentence-transformers (free) or OpenAI (better quality)

### For LLM Response Quality:
- **Gemini:** Free, fast, good for general questions
- **Groq:** Ultra-fast, free, good for summaries
- **GPT-4:** Best reasoning, paid
- **Claude:** Best for long documents, paid

---

## 🐛 Common Issues

### Issue: "Prophet not found"
```bash
pip install prophet --no-cache-dir
# If fails on Windows, install Visual C++ Build Tools first
```

### Issue: "FAISS import error"
```bash
pip uninstall faiss
pip install faiss-cpu
```

### Issue: "Out of memory" with RAG
```python
# Reduce chunk size
rag = FinancialRAGAgent(chunk_size=300)  # Default is 500
```

### Issue: Forecast is flat/wrong
- Check if date column is properly formatted
- Ensure data has at least 40 observations
- Verify target column is numeric
- Try different models (Prophet for seasonality)

---

## 🎓 Learning Path

### Beginner: Start Here
1. Run `app.py` with sample CSV
2. Upload sales data
3. Try basic analysis
4. Ask AI questions in chat

### Intermediate: Explore Features
1. Run `app_extended.py`
2. Try multi-CSV merge
3. Test advanced forecasting
4. Analyze monetary data

### Advanced: Programmatic Use
1. Import agents in Python
2. Build custom pipelines
3. Integrate RAG
4. Create custom reports

---

## 📞 Get Help

**Error Messages:**
- Check terminal output for detailed traceback
- Verify all dependencies installed: `pip list`
- Test with minimal example first

**Performance Issues:**
- Use smaller datasets (<100K rows) for testing
- Disable Prophet if slow (enable_prophet=False)
- Reduce RAG chunk size

**Questions:**
- Check IMPLEMENTATION_GUIDE.md for details
- Review code comments in agent files
- Test individual modules separately

---

## ✅ Checklist Before Running

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] API key set (env var, secrets, or manual)
- [ ] Sample data prepared (CSV with Date + numeric columns)
- [ ] Port 8501 available (Streamlit default)

**You're Ready! 🚀**

```bash
streamlit run app.py
# OR
streamlit run app_extended.py
```

Open browser: `http://localhost:8501`

Enjoy your AI Data Intelligence Agent! 🎉
