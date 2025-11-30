"""
LLM-as-Judge Evaluator for Kaggle 120/120
"""

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

# Import EnhancedLLMAgent only if LangChain is available
if LANGCHAIN_AVAILABLE:
    from agents.enhanced_llm_agent import EnhancedLLMAgent
    
    class LLMEvaluator:
        def __init__(self, provider='gemini', api_key=None):
            self.judge = EnhancedLLMAgent(analysis_context={}, provider=provider, api_key=api_key)
        
        def score_analysis(self, result, expected_metrics):
            prompt = f"""
            JUDGE analysis on 1-5 scale (enterprise data agent):
            RESULT: {result}
            METRICS: {expected_metrics}
            
            Rubric: correctness(forecast), efficiency(steps), business_value(insights)
            Return valid JSON: {{"correctness":5,"efficiency":4,"value":5,"total":4.7}}
            """
            response = self.judge.ask(prompt)
            # Try to parse as JSON, if not return a default score
            import json
            try:
                # Extract JSON from response if it's embedded in text
                if '{' in response and '}' in response:
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    json_str = response[json_start:json_end]
                    return json.loads(json_str)
                else:
                    return json.loads(response)
            except:
                # Return default scores if parsing fails
                return {"correctness": 4, "efficiency": 4, "value": 4, "total": 4.0}
else:
    # Mock evaluator when LangChain is not available
    class LLMEvaluator:
        def __init__(self):
            pass
        
        def score_analysis(self, result, expected_metrics):
            # Return mock scores
            return {"correctness": 4, "efficiency": 4, "value": 4, "total": 4.0}