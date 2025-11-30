"""
Orchestrator Agent - ReAct Pattern for Kaggle 120/120
"""
from agents.kaggle_wrapper import KaggleTracer

# Check if LangChain is available
try:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_groq import ChatGroq
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Define mock LLM agent when LangChain is not available
class MockLLMAgent:
    """Mock LLM agent for when LangChain is not available"""
    def __init__(self, analysis_context=None, provider=None, api_key=None):
        self.context = analysis_context or {}
        self.provider = provider or "gemini"
    
    def ask(self, question):
        """Mock ask method that returns a simple response"""
        return f"Orchestration response for: {question}"

# Import EnhancedLLMAgent only if LangChain is available
if LANGCHAIN_AVAILABLE:
    from agents.enhanced_llm_agent import EnhancedLLMAgent
    from agents.agent_coordinator import AgentCoordinator
    LLM_AGENT_CLASS = EnhancedLLMAgent
    COORDINATOR_CLASS = AgentCoordinator
    
    class OrchestratorAgent:
        def __init__(self, provider="gemini", api_key=None):
            # Use the appropriate LLM agent class
            self.llm = LLM_AGENT_CLASS(analysis_context={}, provider=provider, api_key=api_key)
            self.coordinator = COORDINATOR_CLASS(provider=provider, api_key=api_key)
            
        @KaggleTracer.trace
        def orchestrate(self, df, task="full_analysis", date_col=None, target_col=None):
            """ReAct: Plan → Delegate → Validate → Summarize"""
            plan_prompt = f"""
            You are an orchestration agent for enterprise data.
            
            Dataset shape: {df.shape}
            Columns: {list(df.columns)}
            
            Date column: {date_col}
            Target column: {target_col}
            
            Task: {task}
            
            Plan the analysis: KPIs, trend/seasonality, forecast for target column, and anomaly detection.
            Do NOT say that the dataset is incomplete if target_col is provided.
            """
            plan = self.llm.ask(plan_prompt)
            
            # Delegate to specialist coordinator
            result = self.coordinator.run_full_analysis(df, date_col=date_col, target_col=target_col)
            
            # Validate + summarize
            summary = self.llm.ask(f"Validate + executive summary: {result}")
            
            return {"plan": plan, "analysis": result, "summary": summary}
else:
    # Fallback orchestrator when LangChain is not available
    LLM_AGENT_CLASS = MockLLMAgent
    
    # Simple mock coordinator that directly calls the agents
    class MockCoordinator:
        def __init__(self, provider='groq', api_key=None):
            pass
        
        def run_full_analysis(self, df, date_col=None, target_col=None):
            """Simplified analysis without LLM coordination"""
            return {
                "plan": "Mock analysis plan",
                "cleaned_data_shape": str(df.shape) if hasattr(df, 'shape') else 'N/A',
                "kpis": {"message": "Mock KPIs generated"},
                "forecast_results": [],
                "anomaly_results": {"ensemble": []},
                "analysis_context": {},
                "insights": "Mock insights generated"
            }
    
    class OrchestratorAgent:
        def __init__(self, provider="groq", api_key=None):
            # Use mock classes when LangChain is not available
            self.llm = LLM_AGENT_CLASS(analysis_context={}, provider=provider, api_key=api_key)
            self.coordinator = MockCoordinator(provider=provider, api_key=api_key)
            
        @KaggleTracer.trace
        def orchestrate(self, df, task="full_analysis", date_col=None, target_col=None):
            """ReAct: Plan → Delegate → Validate → Summarize (mock version)"""
            plan_prompt = f"""
            You are an orchestration agent for enterprise data.
            
            Dataset shape: {df.shape}
            Columns: {list(df.columns)}
            
            Date column: {date_col}
            Target column: {target_col}
            
            Task: {task}
            
            Plan the analysis: KPIs, trend/seasonality, forecast for target column, and anomaly detection.
            Do NOT say that the dataset is incomplete if target_col is provided.
            """
            plan = plan_prompt
            
            # Delegate to mock coordinator
            result = self.coordinator.run_full_analysis(df, date_col=date_col, target_col=target_col)
            
            # Simple summary in JSON format
            summary = '{"status": "Analysis completed", "dataset_shape": "%s", "key_results": "%s items processed", "target_column": "%s"}' % (str(df.shape) if hasattr(df.shape, '__iter__') else 'N/A', len(result) if isinstance(result, dict) else 'N/A', target_col or 'Not specified')
            
            return {"plan": plan, "analysis": result, "summary": summary}