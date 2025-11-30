"""
LLM Agent - 100% Working Version with Gemini Support
✅ Works with Groq, Gemini, OpenAI, Anthropic
✅ No function calling errors
✅ Simple and reliable
"""

import os
from typing import Dict, List

try:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_groq import ChatGroq
    from langchain_google_genai import ChatGoogleGenerativeAI  # ✅ GEMINI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# ================== CONFIGURATION ==================
DEFAULT_LLM_PROVIDER = 'groq'
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2048

DEFAULT_MODEL = {
    'openai': 'gpt-4-turbo-preview',
    'anthropic': 'claude-3-5-sonnet-20241022',
    'groq': 'llama-3.3-70b-versatile',
    'gemini': 'models/gemini-2.0-flash-exp'  # ✅ Add "models/" prefix

}

class LLMAgent:
    """LLM Agent for data analysis Q&A"""
    
    def __init__(self, analysis_context: Dict, provider: str = None, api_key: str = None):
        """
        Initialize LLM Agent
        
        Args:
            analysis_context: Dict with KPIs, forecasts, anomalies
            provider: 'openai', 'anthropic', 'groq', or 'gemini'
            api_key: API key
        """
        self.context = analysis_context
        self.provider = (provider or DEFAULT_LLM_PROVIDER).lower()
        self.api_key = api_key
        self.chat_history = []
        
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not installed")
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Create prompt with embedded context
        self.prompt = self._create_prompt()
        
        # Create chain
        self.chain = self.prompt | self.llm | StrOutputParser()
    
    def _initialize_llm(self):
        """Initialize LLM based on provider"""
        
        if self.provider == 'openai':
            api_key = self.api_key or os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OpenAI API key required")
            return ChatOpenAI(
                model=DEFAULT_MODEL['openai'],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                api_key=api_key
            )
        
        elif self.provider == 'anthropic':
            api_key = self.api_key or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError("Anthropic API key required")
            return ChatAnthropic(
                model=DEFAULT_MODEL['anthropic'],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                api_key=api_key
            )
        
        elif self.provider == 'groq':
            api_key = self.api_key or os.getenv('GROQ_API_KEY')
            if not api_key:
                raise ValueError("Groq API key required")
            return ChatGroq(
                model=DEFAULT_MODEL['groq'],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                api_key=api_key
            )
        
        elif self.provider == 'gemini':  # ✅ GEMINI SUPPORT
            api_key = self.api_key or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("Google Gemini API key required")
            return ChatGoogleGenerativeAI(
                model=DEFAULT_MODEL['gemini'],
                temperature=LLM_TEMPERATURE,
                max_output_tokens=LLM_MAX_TOKENS,
                google_api_key=api_key
            )
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _create_prompt(self):
        """Create prompt with analysis context embedded"""
        context_text = self._format_analysis_context()
        
        system_prompt = f"""You are an expert data analyst AI assistant. You have analyzed a dataset and here are the results:

{context_text}

Instructions:
- Answer questions based ONLY on the analysis data above
- Provide specific numbers from the data
- Be clear, concise, and professional
- If asked about something not in the data, say "This information is not available"
- Format large numbers with commas (e.g., 1,234.56)
- Use $ for monetary values"""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
    
    def _format_analysis_context(self) -> str:
        """Format all analysis data into readable text"""
        sections = []
        
        # === DATASET INFO ===
        if 'kpis' in self.context and 'dataset_info' in self.context['kpis']:
            info = self.context['kpis']['dataset_info']
            sections.append(f"""
═══════════════════════════════════════
📋 DATASET INFORMATION
═══════════════════════════════════════
Total Rows: {info.get('rows', 0):,}
Total Columns: {info.get('cols', 0)}
Target Column: {info.get('target_column', 'N/A')}
Domain: {info.get('domain', 'general').title()}
""")
        
        # === KEY METRICS ===
        if 'kpis' in self.context and 'target_metrics' in self.context['kpis']:
            metrics = self.context['kpis']['target_metrics']
            sections.append(f"""
═══════════════════════════════════════
📊 KEY PERFORMANCE METRICS
═══════════════════════════════════════
Total: ${metrics.get('total', 0):,.2f}
Average (Mean): ${metrics.get('mean', 0):,.2f}
Median: ${metrics.get('median', 0):,.2f}
Standard Deviation: ${metrics.get('std', 0):,.2f}
Minimum: ${metrics.get('min', 0):,.2f}
Maximum: ${metrics.get('max', 0):,.2f}
25th Percentile: ${metrics.get('q25', 0):,.2f}
75th Percentile: ${metrics.get('q75', 0):,.2f}
""")
        
        # === TOP CATEGORIES ===
        if 'top_categories' in self.context:
            cats = self.context['top_categories']
            if cats:
                sections.append("\n═══════════════════════════════════════")
                sections.append("🏆 TOP PERFORMERS BY CATEGORY")
                sections.append("═══════════════════════════════════════")
                
                for cat_name, cat_data in list(cats.items())[:5]:
                    if isinstance(cat_data, dict) and 'error' not in cat_data:
                        sections.append(f"\n{cat_name}:")
                        for i, (item, value) in enumerate(list(cat_data.items())[:10], 1):
                            sections.append(f"  {i}. {item}: ${value:,.2f}")
        
        # === FORECAST ===
        if 'forecast' in self.context and self.context['forecast']:
            forecast_list = self.context['forecast']
            if forecast_list:
                total_forecast = sum(f['prediction'] for f in forecast_list)
                avg_forecast = total_forecast / len(forecast_list)
                
                sections.append(f"""
═══════════════════════════════════════
📈 FORECAST (Next 14 Days)
═══════════════════════════════════════
Total Predicted: ${total_forecast:,.2f}
Daily Average: ${avg_forecast:,.2f}
Trend: {self.context.get('growth_rate', 0):+.2f}% growth
""")
                
                sections.append("\nDaily Predictions:")
                for i, f in enumerate(forecast_list[:14], 1):
                    sections.append(f"  Day {i} ({f.get('date', 'N/A')}): ${f.get('prediction', 0):,.2f}")
        
        # === ANOMALIES ===
        if 'anomalies' in self.context and self.context['anomalies']:
            anomalies = self.context['anomalies']
            if anomalies:
                critical = [a for a in anomalies if a.get('severity') == 'CRITICAL']
                high = [a for a in anomalies if a.get('severity') == 'HIGH']
                medium = [a for a in anomalies if a.get('severity') == 'MEDIUM']
                
                sections.append(f"""
═══════════════════════════════════════
🚨 ANOMALY DETECTION RESULTS
═══════════════════════════════════════
Total Anomalies Detected: {len(anomalies)}
Critical: {len(critical)}
High: {len(high)}
Medium: {len(medium)}
""")
                
                if critical:
                    sections.append("\nCritical Anomalies:")
                    for i, a in enumerate(critical[:5], 1):
                        sections.append(f"  {i}. Date: {a.get('date', 'N/A')}, Value: ${a.get('value', 0):,.2f}")
        
        # === GROWTH ANALYSIS ===
        if 'growth_rate' in self.context:
            growth = self.context['growth_rate']
            sections.append(f"""
═══════════════════════════════════════
📊 GROWTH ANALYSIS
═══════════════════════════════════════
Overall Growth Rate: {growth:+.2f}%
""")
        
        return "\n".join(sections)
    
    def ask(self, question: str) -> str:
        """
        Ask a question about the analysis
        
        Args:
            question: User's question
            
        Returns:
            AI-generated answer
        """
        try:
            response = self.chain.invoke({"question": question})
            
            # Store in history
            self.chat_history.append({
                'question': question,
                'answer': response
            })
            
            return response
        
        except Exception as e:
            error_msg = f"I encountered an error: {str(e)}"
            self.chat_history.append({
                'question': question,
                'answer': error_msg
            })
            return error_msg
    
    def get_chat_history(self) -> List[Dict]:
        """Get chat history"""
        return self.chat_history
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []
