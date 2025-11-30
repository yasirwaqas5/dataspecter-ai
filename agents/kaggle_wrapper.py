"""
Kaggle Wrapper - Safe modules for 120/120 features
Non-breaking addition to existing app logic
"""
import os
import sys
import time
import json
import uuid
from typing import Dict, Any

# Add current directory to path
sys.path.insert(0, '.')

# Safe config import
try:
    from config import KAGGLE_MODE, USE_MULTI_AGENT, USE_TRACING
except ImportError:
    KAGGLE_MODE = False
    USE_MULTI_AGENT = False
    USE_TRACING = False

class KaggleTracer:
    """Lightweight tracing for Kaggle competition metrics"""
    
    @staticmethod
    def trace(func):
        """Decorator to trace any function execution"""
        # If tracing is disabled, return the function as-is
        if not USE_TRACING:
            return func
            
        def wrapper(*args, **kwargs):
            # Generate unique trace ID
            trace_id = str(uuid.uuid4())[:8]
            start = time.time()
            print(f"🚀 [{trace_id}] {func.__name__}")
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                duration = time.time() - start
                print(f"✅ [{trace_id}] SUCCESS ({duration:.1f}s)")
                KaggleTracer.save_trace(trace_id, "SUCCESS", duration, result)
                return result
                
            except Exception as e:
                duration = time.time() - start
                print(f"❌ [{trace_id}] ERROR ({duration:.1f}s): {e}")
                KaggleTracer.save_trace(trace_id, "FAILED", duration, str(e))
                raise
        
        return wrapper
    
    @staticmethod
    def save_trace(trace_id: str, status: str, duration: float, result: Any):
        """
        Save trace to file for leaderboard metrics
        
        Args:
            trace_id: Unique identifier
            status: SUCCESS/FAILED
            duration: Execution time in seconds
            result: Function result (truncated)
        """
        # Create traces directory
        os.makedirs(".traces", exist_ok=True)
        
        # Truncate result for storage
        result_str = str(result)
        if len(result_str) > 200:
            result_str = result_str[:200] + "..."
        
        # Create trace record
        trace = {
            "id": trace_id,
            "status": status,
            "duration": duration,
            "result": result_str,
            "timestamp": time.time()
        }
        
        # Save to file
        with open(f".traces/{trace_id}.json", "w") as f:
            json.dump(trace, f, indent=2)

def kaggle_metrics():
    """Display Kaggle leaderboard metrics"""
    try:
        # Check if traces directory exists
        if not os.path.exists(".traces"):
            return "No traces yet. Run analysis!"
        
        # Get all trace files
        traces = [f for f in os.listdir(".traces") if f.endswith('.json')]
        if not traces:
            return "No traces yet. Run analysis!"
        
        # Calculate success rate
        success_count = 0
        total_duration = 0
        
        for trace_file in traces:
            try:
                with open(f".traces/{trace_file}", "r") as f:
                    trace_data = json.load(f)
                    if trace_data.get("status") == "SUCCESS":
                        success_count += 1
                    total_duration += trace_data.get("duration", 0)
            except Exception:
                continue
        
        success_rate = success_count / len(traces) if traces else 0
        avg_duration = total_duration / len(traces) if traces else 0
        
        return f"🏆 Traces: {len(traces)} | Success: {success_rate:.0%} | Multi-Agent: {'✅' if USE_MULTI_AGENT else '📊'} | Avg Time: {avg_duration:.1f}s"
        
    except Exception as e:
        return f"Kaggle metrics error: {e}"