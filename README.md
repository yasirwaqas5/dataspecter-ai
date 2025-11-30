# Dataspecter AI – AI Data Intelligence Agent

[![Kaggle 120/120](https://img.shields.io/badge/Kaggle-120%2F120-brightgreen)](https://kaggle.com)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 🎯 Overview

Dataspecter AI is an intelligent data analysis platform that transforms hours of manual data analysis into seconds of automated insights. Leveraging multi-agent orchestration and advanced LLM capabilities, it provides comprehensive data intelligence across various domains.

## ✨ Key Features

- **Multi-Agent Orchestration**: Coordinated analysis pipeline with 6 specialized agents
- **Single CSV Analysis**: Automated KPI computation, forecasting, and anomaly detection
- **Multi-CSV Merge**: Intelligent schema alignment and data fusion across multiple files
- **Monetary Analysis**: Specialized economic indicators analysis (M1/M3/CPI)
- **Document RAG**: PDF/DOCX/TXT analysis with semantic search and Q&A
- **Natural Language Q&A**: Chat with your data using advanced LLMs
- **Multiple LLM Support**: Groq (primary) with Gemini/OpenAI/Anthropic options

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/yasirwaqas5/dataspecter-ai.git
cd dataspecter-ai

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
# source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set an LLM API key (recommended: Groq)
# Windows (PowerShell)
$env:GROQ_API_KEY="your_groq_key_here"
# macOS / Linux
# export GROQ_API_KEY="your_groq_key_here"

# 5. Run application
streamlit run app.py
```

## ▶️ Using the App

Open the app in your browser (usually http://localhost:8501).

Choose a mode from the sidebar:

- **Single Dataset Analysis**
- **Multi‑CSV Merge**
- **Monetary Analysis**
- **Document RAG**
- **Orchestrator / AI Assistant**

Steps:
1. Upload the appropriate file(s) for that mode
2. Configure columns (date/target) where required
3. Click the action button (e.g. "Run Analysis", "Merge & Analyze", "Process Documents")
4. Explore charts, tables, and AI‑generated insights

## 📁 Supported File Types

- **Data Files**: CSV (primary), XLSX/XLS, JSON, Parquet
- **Document Files**: TXT, PDF, DOCX for RAG mode

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Forecasting / Time Series**: Statsmodels, Prophet, pmdarima
- **ML / Metrics**: Scikit‑learn
- **Visualization**: Plotly
- **LLMs**: Groq (primary), optional Gemini / OpenAI / Anthropic via LangChain
- **RAG**: Simple text chunking + LLM QA (no heavy vector DB required for demo)

## ☁️ Deployment

- **Local**: `streamlit run app.py`
- **Streamlit Cloud**: Connect this GitHub repo and set secrets (e.g. GROQ_API_KEY) in the Streamlit "Secrets" UI
- **Docker / Cloud Run**: See [DEPLOYMENT.md](DEPLOYMENT.md) for containerization and cloud steps

## 📚 Additional Docs

- [ARCHITECTURE.md](ARCHITECTURE.md) – High‑level system and agent design
- [DEPLOYMENT.md](DEPLOYMENT.md) – Detailed deployment instructions
- [DEMO_SCRIPT.md](DEMO_SCRIPT.md) – Suggested narration for the demo video
- [FEATURES_MATRIX.md](FEATURES_MATRIX.md) / [TECH_STACK.md](TECH_STACK.md) – Extended feature and stack details

## 📄 License

This project is for the Google Kaggle AI Agents Intensive Capstone 2025. Licensed under the MIT License - see the [LICENSE](LICENSE) file for details.