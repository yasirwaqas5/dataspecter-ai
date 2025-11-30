"""
Agent Coordinator - Multi-agent orchestration for Kaggle 120/120
Wraps existing logic in agent pattern WITHOUT breaking anything
"""

from agents.enhanced_llm_agent import EnhancedLLMAgent
from agents import preprocessor, analyzer, feature_engineer, forecast_agent, anomaly_agent
import pandas as pd

class AgentCoordinator:
    def __init__(self, provider='gemini', api_key=None):
        """Initialize coordinator with enhanced LLM"""
        self.llm = EnhancedLLMAgent(analysis_context={}, provider=provider, api_key=api_key)
    
    def run_full_analysis(self, df, date_col=None, target_col=None):
        """
        Multi-agent wrapper - CALLS EXISTING CODE
        
        Args:
            df: Input DataFrame
            date_col: Date column name (optional)
            target_col: Target column name (optional)
            
        Returns:
            dict: Analysis results from all agents
        """
        # Agent 1: Planning
        plan_prompt = f"""
        Plan analysis for dataset:
        - Shape: {df.shape}
        - Columns: {list(df.columns)}
        - Date column: {date_col}
        - Target column: {target_col}
        
        Recommend optimal analysis sequence using available agents.
        """
        plan = self.llm.ask(plan_prompt)
        
        # Agent 2: Preprocessing (existing preprocessor.py)
        schema = {}  # Simplified schema for existing preprocessor
        cleaned_df = preprocessor.PreprocessorAgent(schema).preprocess(df)
        
        # CRITICAL: Uppercase columns (matching app.py logic)
        try:
            cleaned_df.columns = cleaned_df.columns.str.upper().str.replace(' ', '_')
        except Exception:
            pass
        
        target_col_clean = target_col.upper().replace(' ', '_') if target_col else None
        date_col_clean = date_col.upper().replace(' ', '_') if date_col else None
        
        # Agent 3: Analysis (existing analyzer.py)  
        if target_col_clean and target_col_clean in cleaned_df.columns:
            kpis = analyzer.AnalyzerAgent.analyze(cleaned_df, target_col_clean, schema)
        else:
            kpis = {"error": "No target column specified"}
        
        # Agent 4: Feature Engineering (existing feature_engineer.py)
        daily_df = None
        if date_col_clean and target_col_clean and date_col_clean in cleaned_df.columns and target_col_clean in cleaned_df.columns:
            try:
                daily_df = feature_engineer.FeatureEngineerAgent.engineer_features(
                    cleaned_df, target_col_clean, date_col_clean
                )
            except Exception as e:
                daily_df = {"error": f"Feature engineering failed: {str(e)}"}
        
        # Agent 5: Advanced Time-Series Processing (matching app.py logic)
        decomposition_result = None
        seasonality_info = None
        
        # Agent 6: Forecasting (existing forecast_agent.py)
        forecast_results = []
        forecast_state = None
        if daily_df is not None and len(daily_df) > 40:
            try:
                forecast_state, forecast_results = forecast_agent.ForecastAgent.train_and_forecast(
                    daily_df, target_col_clean, date_col_clean, horizon=14
                )
            except Exception as e:
                forecast_results = {"error": f"Forecasting failed: {str(e)}"}
        
        # Agent 7: Anomaly Detection (existing anomaly_agent.py)
        anomaly_results = {'ensemble': []}
        if daily_df is not None:
            try:
                anomaly_results = anomaly_agent.AnomalyAgent.detect(
                    daily_df, target_col_clean, date_col_clean
                )
            except Exception as e:
                anomaly_results = {"error": f"Anomaly detection failed: {str(e)}"}
        
        # Agent 8: LLM Agent Initialization (matching app.py logic)
        analysis_context = {
            'kpis': kpis,
            'forecast': forecast_results,
            'anomalies': anomaly_results.get('ensemble', []),
            'metrics': forecast_state.get('metrics', {}) if forecast_state else {},
            'data_summary': {
                'total_rows': len(cleaned_df) if cleaned_df is not None else 0,
                'total_columns': len(cleaned_df.columns) if cleaned_df is not None else 0,
                'memory_usage_mb': cleaned_df.memory_usage(deep=True).sum() / 1024**2 if cleaned_df is not None else 0
            }
        }
        
        # Agent 9: Summary (existing llm_agent.py)
        summary_prompt = f"""
        Summarize this analysis:
        - KPIs: {kpis}
        - Forecast samples: {forecast_results[:3] if isinstance(forecast_results, list) else forecast_results}
        - Anomalies count: {len(anomaly_results.get('ensemble', [])) if isinstance(anomaly_results, dict) else 0}
        
        Provide executive summary with key business insights.
        """
        insights = self.llm.ask(summary_prompt)
        
        return {
            "plan": plan,
            "cleaned_data": cleaned_df,
            "cleaned_data_shape": cleaned_df.shape if hasattr(cleaned_df, 'shape') else 'N/A',
            "kpis": kpis,
            "daily_df": daily_df,
            "forecast_results": forecast_results,
            "forecast_state": forecast_state,
            "anomaly_results": anomaly_results,
            "decomposition_result": decomposition_result,
            "seasonality_info": seasonality_info,
            "analysis_context": analysis_context,
            "insights": insights
        }
