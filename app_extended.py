"""
AI Data Intelligence Agent v7.0 - Extended Edition
New features:
- Advanced time-series analysis with decomposition
- ARIMA, Prophet, and auto-selection forecasting
- Monetary aggregates analysis (M1/M3/CPI)
- RAG for document analysis
- Multi-CSV merging support
"""

import streamlit as st

st.set_page_config(
    page_title="AI Data Intelligence Agent v7.0",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime
from pathlib import Path
import sys
import traceback
import os

sys.path.insert(0, str(Path(__file__).parent))

# Import all agents
from agents.preprocessor import PreprocessorAgent
from agents.analyzer import AnalyzerAgent
from agents.feature_engineer import FeatureEngineerAgent
from agents.forecast_agent import ForecastAgent
from agents.llm_agent import LLMAgent
from agents.timeseries_processor import TimeSeriesProcessor
from agents.advanced_forecast_agent import AdvancedForecastAgent
from agents.monetary_aggregates import MonetaryAggregatesAnalyzer
from agents.rag_agent import FinancialRAGAgent

# ========== SESSION STATE ==========
def init_session_state():
    defaults = {
        'data_loaded': False,
        'df_raw': None,
        'df_clean': None,
        'schema': None,
        'analysis_complete': False,
        'forecast_results': None,
        'advanced_forecast_results': None,
        'monetary_analysis': None,
        'rag_agent': None,
        'rag_enabled': False,
        'chat_history': [],
        'llm_agent': None,
        'llm_provider': 'gemini',
        'api_key_input': '',
        'multi_csv_mode': False,
        'loaded_datasets': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ========== SIDEBAR ==========
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mode selection
        st.subheader("📁 Data Mode")
        mode = st.radio(
            "Select Mode",
            ["Single Dataset", "Multi-CSV Merge", "Monetary Analysis", "Document RAG"],
            help="Choose your analysis mode"
        )
        
        st.session_state.multi_csv_mode = (mode == "Multi-CSV Merge")
        
        # LLM Provider
        st.markdown("---")
        st.subheader("🧠 LLM Provider")
        llm_provider = st.selectbox(
            "Provider",
            ['gemini', 'groq', 'openai', 'anthropic'],
            index=['gemini', 'groq', 'openai', 'anthropic'].index(st.session_state.llm_provider)
        )
        st.session_state.llm_provider = llm_provider
        
        # API Key
        api_key = st.text_input(
            "API Key (Optional)",
            type="password",
            value=st.session_state.get('api_key_input', ''),
            help="Enter API key or leave blank to use environment variables"
        )
        if api_key:
            st.session_state.api_key_input = api_key
            st.success("✅ API Key Set")
        
        st.markdown("---")
        
        return mode

# ========== MAIN APP ==========
def main():
    init_session_state()
    
    st.title("🤖 AI Data Intelligence Agent v7.0")
    st.markdown("**Extended Edition** - Advanced Analytics & RAG Support")
    st.markdown("---")
    
    mode = render_sidebar()
    
    # ========== SINGLE DATASET MODE ==========
    if mode == "Single Dataset":
        render_single_dataset_mode()
    
    # ========== MULTI-CSV MERGE MODE ==========
    elif mode == "Multi-CSV Merge":
        render_multi_csv_mode()
    
    # ========== MONETARY ANALYSIS MODE ==========
    elif mode == "Monetary Analysis":
        render_monetary_analysis_mode()
    
    # ========== DOCUMENT RAG MODE ==========
    elif mode == "Document RAG":
        render_rag_mode()

# ========== SINGLE DATASET MODE ==========
def render_single_dataset_mode():
    st.header("📊 Single Dataset Analysis")
    
    uploaded_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file:
        if st.button("🚀 Load & Analyze"):
            with st.spinner("Loading..."):
                try:
                    # Load file
                    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                    st.session_state.df_raw = df
                    st.session_state.data_loaded = True
                    
                    st.success(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")
                    
                    # Auto-detect schema
                    from agents.preprocessor import PreprocessorAgent
                    schema = detect_schema(df)
                    st.session_state.schema = schema
                    
                    # Show preview
                    st.dataframe(df.head(10), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error: {e}")
    
    if st.session_state.data_loaded:
        st.markdown("---")
        st.subheader("🎯 Configure Analysis")
        
        df = st.session_state.df_raw
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        target_col = st.selectbox("Target Variable", numeric_cols)
        date_col = st.selectbox("Date Column (Optional)", [None] + list(df.columns))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            enable_advanced_forecast = st.checkbox("Advanced Forecasting (ARIMA/Prophet)", value=True)
        
        with col2:
            enable_decomposition = st.checkbox("Time-Series Decomposition", value=True)
        
        with col3:
            enable_seasonality = st.checkbox("Seasonality Detection", value=True)
        
        if st.button("🚀 Run Analysis", type="primary"):
            run_extended_analysis(
                df, target_col, date_col,
                enable_advanced_forecast,
                enable_decomposition,
                enable_seasonality
            )

# ========== MULTI-CSV MODE ==========
def render_multi_csv_mode():
    st.header("📂 Multi-CSV Merge & Analysis")
    
    st.info("Upload multiple CSV files to merge and analyze together")
    
    uploaded_files = st.file_uploader(
        "Upload Multiple CSVs",
        type=['csv'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"📁 {len(uploaded_files)} files uploaded")
        
        merge_column = st.text_input("Merge Column (Common column across all files)", "Date")
        merge_type = st.selectbox("Merge Type", ["outer", "inner", "left", "right"])
        
        if st.button("Merge & Analyze"):
            with st.spinner("Merging datasets..."):
                try:
                    dfs = []
                    for file in uploaded_files:
                        df = pd.read_csv(file)
                        df['source'] = file.name
                        dfs.append(df)
                    
                    # Merge all
                    merged_df = dfs[0]
                    for df in dfs[1:]:
                        merged_df = pd.merge(merged_df, df, on=merge_column, how=merge_type, suffixes=('', '_dup'))
                    
                    st.session_state.df_raw = merged_df
                    st.session_state.data_loaded = True
                    
                    st.success(f"✅ Merged into {len(merged_df):,} rows × {len(merged_df.columns)} columns")
                    st.dataframe(merged_df.head(10), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Merge failed: {e}")

# ========== MONETARY ANALYSIS MODE ==========
def render_monetary_analysis_mode():
    st.header("💰 Monetary Aggregates Analysis")
    
    st.markdown("""
    Analyze macroeconomic indicators:
    - Money Supply (M1, M2, M3)
    - CPI / Inflation
    - Interest Rates
    - YOY/MOM Growth
    - Correlation Analysis
    """)
    
    uploaded_file = st.file_uploader("Upload Monetary Dataset (CSV/Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        date_col = st.selectbox("Date Column", df.columns)
        value_cols = st.multiselect("Value Columns (M1, M3, CPI, etc.)", df.columns.tolist())
        
        if st.button("Analyze Monetary Data"):
            with st.spinner("Analyzing..."):
                try:
                    analyzer = MonetaryAggregatesAnalyzer()
                    
                    # Generate comprehensive report
                    report = analyzer.generate_monetary_report(df, date_col, value_cols)
                    
                    st.session_state.monetary_analysis = report
                    
                    # Display results
                    st.success("✅ Analysis Complete")
                    
                    # Summary Statistics
                    st.subheader("📊 Summary Statistics")
                    for var, stats in report.get('summary_statistics', {}).items():
                        st.markdown(f"**{var}**")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Current", f"{stats['current']:,.2f}")
                        col2.metric("Mean", f"{stats['mean']:,.2f}")
                        col3.metric("Min", f"{stats['min']:,.2f}")
                        col4.metric("Max", f"{stats['max']:,.2f}")
                    
                    # Growth Analysis
                    st.markdown("---")
                    st.subheader("📈 Growth Analysis")
                    for var, growth in report.get('growth_analysis', {}).items():
                        col1, col2 = st.columns(2)
                        col1.metric(f"{var} - YOY Growth", f"{growth['latest_yoy']:.2f}%")
                        col2.metric(f"{var} - MOM Growth", f"{growth['latest_mom']:.2f}%")
                    
                    # Correlation Matrix
                    if 'correlation_analysis' in report and 'correlation_matrix' in report['correlation_analysis']:
                        st.markdown("---")
                        st.subheader("🔗 Correlation Matrix")
                        corr_df = pd.DataFrame(report['correlation_analysis']['correlation_matrix'])
                        
                        fig = px.imshow(
                            corr_df,
                            text_auto='.2f',
                            aspect='auto',
                            color_continuous_scale='RdBu_r',
                            title="Correlation Heatmap"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Top Insights
                    if 'insights' in report and report['insights']:
                        st.markdown("---")
                        st.subheader("💡 Key Insights")
                        for insight in report['insights']:
                            st.info(insight)
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.code(traceback.format_exc())

# ========== RAG MODE ==========
def render_rag_mode():
    st.header("📄 Document Analysis with RAG")
    
    st.markdown("""
    Upload financial documents (PDF, TXT, DOCX) to enable AI-powered document Q&A.
    The system will:
    - Extract and chunk documents
    - Create embeddings
    - Enable semantic search
    - Combine with dataset insights
    """)
    
    uploaded_docs = st.file_uploader(
        "Upload Documents",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True
    )
    
    if uploaded_docs:
        st.write(f"📁 {len(uploaded_docs)} documents uploaded")
        
        if st.button("Process Documents"):
            with st.spinner("Processing..."):
                try:
                    # Initialize RAG agent
                    rag_agent = FinancialRAGAgent(embedding_model='sentence-transformers')
                    
                    # Load documents
                    result = rag_agent.load_documents(uploaded_files=uploaded_docs)
                    
                    if result['success']:
                        st.success(f"✅ Loaded {result['documents_loaded']} documents ({result['total_chunks']} chunks)")
                        
                        # Create vector store
                        vs_result = rag_agent.create_vector_store()
                        
                        if vs_result['success']:
                            st.success(f"✅ Created vector store with {vs_result['num_vectors']} vectors")
                            st.session_state.rag_agent = rag_agent
                            st.session_state.rag_enabled = True
                        else:
                            st.error(f"Failed to create vector store: {vs_result.get('error')}")
                    else:
                        st.error(f"Document loading failed")
                        if result.get('errors'):
                            for error in result['errors']:
                                st.error(error)
                    
                except Exception as e:
                    st.error(f"Processing failed: {e}")
                    st.code(traceback.format_exc())
    
    # RAG Query Interface
    if st.session_state.rag_enabled and st.session_state.rag_agent:
        st.markdown("---")
        st.subheader("🔍 Ask Questions About Your Documents")
        
        query = st.text_input("Enter your question:", placeholder="e.g., What is the revenue growth?")
        
        if st.button("Search"):
            if query:
                with st.spinner("Searching..."):
                    results = st.session_state.rag_agent.retrieve(query, top_k=3)
                    
                    st.markdown("### 📚 Relevant Sections:")
                    for result in results:
                        with st.expander(f"Result {result['rank']} - Relevance: {result['relevance_score']:.2f}"):
                            st.markdown(f"**Source:** {result['metadata'].get('source', 'Unknown')}")
                            st.markdown(result['content'])

# ========== EXTENDED ANALYSIS FUNCTION ==========
def run_extended_analysis(df, target_col, date_col, enable_advanced_forecast, enable_decomposition, enable_seasonality):
    with st.spinner("🔬 Running Extended Analysis..."):
        progress = st.progress(0)
        
        try:
            # Clean data
            progress.progress(20)
            preprocessor = PreprocessorAgent({})
            df_clean = preprocessor.preprocess(df)
            
            target_col_clean = target_col.upper().replace(' ', '_')
            
            # Time-series processing
            if date_col and enable_decomposition:
                progress.progress(40)
                st.info("🔄 Time-series decomposition...")
                
                ts_processor = TimeSeriesProcessor()
                date_col_clean = date_col.upper().replace(' ', '_')
                
                # Decompose
                decomp_result = ts_processor.decompose_timeseries(
                    df_clean, date_col_clean, target_col_clean, period=7
                )
                
                if decomp_result.get('success'):
                    st.success("✅ Decomposition complete")
                    st.session_state.decomposition = decomp_result
            
            # Advanced forecasting
            if date_col and enable_advanced_forecast:
                progress.progress(60)
                st.info("🔮 Training advanced forecast models...")
                
                date_col_clean = date_col.upper().replace(' ', '_') if date_col else None
                
                # Prepare data
                ts_df = TimeSeriesProcessor.prepare_timeseries(
                    df_clean, date_col_clean, target_col_clean, freq='D'
                )
                
                # Train all models
                forecast_results = AdvancedForecastAgent.train_all_models(
                    ts_df, date_col_clean, target_col_clean, horizon=14
                )
                
                st.session_state.advanced_forecast_results = forecast_results
                st.success(f"✅ Trained {len(forecast_results['models'])} models")
            
            progress.progress(80)
            
            # Basic analysis
            from agents.analyzer import AnalyzerAgent
            kpis = AnalyzerAgent.analyze(df_clean, target_col_clean, {})
            st.session_state.kpis = kpis
            
            progress.progress(100)
            st.session_state.analysis_complete = True
            st.success("🎉 Analysis Complete!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.code(traceback.format_exc())

# ========== HELPER FUNCTION ==========
def detect_schema(df):
    """Simple schema detection"""
    return {
        'datetime_columns': [],
        'categorical_columns': [],
        'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
        'text_columns': [],
        'id_columns': [],
        'detected_domain': 'general',
        'target_candidates': []
    }

if __name__ == "__main__":
    main()
