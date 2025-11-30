"""
Enhanced LLM Agent with RAG Integration
Combines:
- Dataset insights (KPIs, forecasts, anomalies)
- RAG document retrieval
- Long-context reasoning
- Financial domain expertise
"""

import os
from typing import Dict, List, Optional
from .llm_agent import LLMAgent
from .rag_agent import RAGAgent

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


class EnhancedLLMAgent(LLMAgent):
    """
    Enhanced LLM Agent with RAG and advanced reasoning
    """
    
    def __init__(self, analysis_context: Dict, provider: str = None, 
                 api_key: str = None, rag_agent: Optional[RAGAgent] = None):
        """
        Initialize Enhanced LLM Agent
        
        Args:
            analysis_context: Dataset analysis results
            provider: LLM provider
            api_key: API key
            rag_agent: Optional RAG agent for document retrieval
        """
        super().__init__(analysis_context, provider, api_key)
        self.rag_agent = rag_agent
        self.enable_rag = rag_agent is not None
        
        # Override prompt with enhanced version
        if self.enable_rag:
            self.prompt = self._create_enhanced_prompt()
            self.chain = self.prompt | self.llm | StrOutputParser()
    
    def _create_enhanced_prompt(self):
        """Create enhanced prompt with RAG support"""
        context_text = self._format_analysis_context()
        
        system_prompt = f"""You are an expert financial data analyst AI with access to:
1. **Dataset Analysis Results** (KPIs, forecasts, trends, anomalies)
2. **Financial Documents** (via semantic search when available)

## Dataset Insights:
{context_text}

## Your Capabilities:
- Analyze numerical data and identify trends
- Interpret forecasting results and explain implications
- Correlate dataset insights with document information
- Provide actionable business recommendations
- Explain complex financial concepts clearly

## Instructions:
1. **Answer based on available data** - Use specific numbers and dates
2. **Combine multiple sources** - Link dataset insights with document information when relevant
3. **Be precise and concise** - Avoid hallucination; state "not available" when data is missing
4. **Provide context** - Explain what the numbers mean for business decisions
5. **Use formatting** - Use bullet points, tables, and emphasis for clarity

## Response Format:
- Use **bold** for key metrics
- Use bullet points for lists
- Include $ for monetary values
- Show percentages with % symbol
- Cite sources when using document information

When asked a question, you will:
1. First check if the answer is in the dataset analysis
2. Then check if relevant information is in the documents (if RAG is enabled)
3. Combine both sources for comprehensive answers
4. Provide clear, actionable insights"""

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
    
    def ask(self, question: str, retrieve_docs: bool = True) -> str:
        """
        Ask a question with RAG enhancement
        
        Args:
            question: User's question
            retrieve_docs: Whether to retrieve relevant documents
            
        Returns:
            AI-generated answer
        """
        try:
            # Enhance question with RAG context if available
            enhanced_question = question
            
            if self.enable_rag and retrieve_docs and self.rag_agent:
                # Retrieve relevant documents
                doc_context = self.rag_agent.get_context_for_query(question, max_length=1500)
                
                if doc_context:
                    enhanced_question = f"""Question: {question}

{doc_context}

Based on both the dataset analysis provided in the system context AND the document excerpts above, please answer the question comprehensively."""
            
            # Get response
            response = self.chain.invoke({"question": enhanced_question})
            
            # Store in history
            self.chat_history.append({
                'question': question,
                'answer': response,
                'rag_used': self.enable_rag and retrieve_docs
            })
            
            return response
        
        except Exception as e:
            error_msg = f"I encountered an error: {str(e)}"
            self.chat_history.append({
                'question': question,
                'answer': error_msg,
                'rag_used': False
            })
            return error_msg
    
    def explain_forecast(self) -> str:
        """Generate detailed forecast explanation"""
        if 'forecast' not in self.context or not self.context['forecast']:
            return "No forecast data available."
        
        forecast_list = self.context['forecast']
        total_pred = sum(f['prediction'] for f in forecast_list)
        avg_pred = total_pred / len(forecast_list)
        
        growth_rate = self.context.get('growth_rate', 0)
        
        explanation = f"""
## 📈 Forecast Explanation

**Forecast Period:** Next {len(forecast_list)} days

**Key Metrics:**
- Total Predicted Value: **${total_pred:,.2f}**
- Daily Average: **${avg_pred:,.2f}**
- Growth Trend: **{growth_rate:+.2f}%**

**What This Means:**
"""
        
        if growth_rate > 5:
            explanation += f"- Strong **upward trend** detected ({growth_rate:.1f}% growth)\n"
            explanation += "- Business is expanding, consider scaling operations\n"
        elif growth_rate < -5:
            explanation += f"- **Declining trend** detected ({growth_rate:.1f}% decrease)\n"
            explanation += "- Review cost structure and market conditions\n"
        else:
            explanation += "- **Stable trend** - minimal growth or decline expected\n"
        
        # Model performance
        if 'metrics' in self.context and self.context['metrics']:
            metrics = self.context['metrics']
            if 'rmse' in metrics:
                explanation += f"\n**Model Accuracy:**\n"
                explanation += f"- RMSE: {metrics.get('rmse', 0):.2f}\n"
                explanation += f"- MAE: {metrics.get('mae', 0):.2f}\n"
        
        return explanation
    
    def explain_anomalies(self) -> str:
        """Generate detailed anomaly explanation"""
        if 'anomalies' not in self.context or not self.context['anomalies']:
            return "No anomalies detected."
        
        anomalies = self.context['anomalies']
        critical = [a for a in anomalies if a.get('severity') == 'CRITICAL']
        high = [a for a in anomalies if a.get('severity') == 'HIGH']
        
        explanation = f"""
## 🚨 Anomaly Detection Report

**Total Anomalies:** {len(anomalies)}
- Critical: **{len(critical)}**
- High: **{len(high)}**

**What Are Anomalies?**
Anomalies are unusual data points that deviate significantly from normal patterns. They may indicate:
- Data quality issues
- Business events (promotions, holidays)
- System errors
- Market disruptions

**Critical Anomalies:**
"""
        
        for i, anom in enumerate(critical[:3], 1):
            explanation += f"{i}. Date: **{anom['date']}**, Value: **${anom['value']:,.2f}**\n"
        
        explanation += "\n**Recommended Actions:**\n"
        explanation += "1. Investigate root causes of critical anomalies\n"
        explanation += "2. Validate data accuracy for flagged dates\n"
        explanation += "3. Document business events that explain unusual patterns\n"
        
        return explanation
    
    def generate_executive_summary(self) -> str:
        """Generate comprehensive executive summary"""
        kpis = self.context.get('kpis', {})
        metrics = kpis.get('target_metrics', {})
        
        summary = f"""
# 📊 Executive Summary

## Key Performance Indicators
- **Total Value:** ${metrics.get('total', 0):,.2f}
- **Average:** ${metrics.get('mean', 0):,.2f}
- **Median:** ${metrics.get('median', 0):,.2f}
- **Range:** ${metrics.get('min', 0):,.2f} to ${metrics.get('max', 0):,.2f}

## Forecast Outlook
"""
        
        if 'forecast' in self.context and self.context['forecast']:
            growth = self.context.get('growth_rate', 0)
            summary += f"- Projected Growth: **{growth:+.2f}%**\n"
            
            if growth > 0:
                summary += "- **Positive outlook** - Growth expected\n"
            else:
                summary += "- **Caution** - Declining trend detected\n"
        
        summary += "\n## Risk Assessment\n"
        
        anomalies = self.context.get('anomalies', [])
        if len(anomalies) > 10:
            summary += "- ⚠️ High number of anomalies detected - investigate data quality\n"
        elif len(anomalies) > 0:
            summary += f"- ℹ️ {len(anomalies)} anomalies detected - within normal range\n"
        else:
            summary += "- ✅ No significant anomalies - data quality good\n"
        
        summary += "\n## Recommended Actions\n"
        summary += "1. Monitor forecast accuracy over next reporting period\n"
        summary += "2. Review anomalous data points for business insights\n"
        summary += "3. Update forecasts monthly with new data\n"
        
        return summary
    
    def ask_with_chain_of_thought(self, question: str) -> str:
        """
        Answer question with explicit reasoning steps
        
        Args:
            question: Complex question requiring reasoning
            
        Returns:
            Detailed answer with reasoning steps
        """
        cot_prompt = f"""Question: {question}

Please answer this question using step-by-step reasoning:

1. **Identify**: What data is relevant to this question?
2. **Analyze**: What patterns or insights can be extracted?
3. **Synthesize**: How do different data points relate?
4. **Conclude**: What is the final answer with supporting evidence?

Provide your response in this structured format."""
        
        return self.ask(cot_prompt, retrieve_docs=True)
    
    def compare_with_document(self, dataset_insight: str) -> str:
        """
        Compare dataset insight with document information
        
        Args:
            dataset_insight: Insight from dataset (e.g., "Revenue increased 20%")
            
        Returns:
            Comparison and context from documents
        """
        if not self.enable_rag:
            return "RAG not enabled - cannot compare with documents"
        
        comparison_question = f"""
Dataset shows: {dataset_insight}

Do the financial documents provide any context, explanations, or additional information about this?
Compare what the dataset shows with what the documents explain."""
        
        return self.ask(comparison_question, retrieve_docs=True)
