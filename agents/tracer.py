"""
Simple Tracer - Drop-in observability for Kaggle 120/120
Works with ANY function - zero dependencies
"""

import time, json, os, uuid
from functools import wraps

# Create traces directory
os.makedirs(".traces", exist_ok=True)

class SimpleTracer:
    """Lightweight tracing for Kaggle competition metrics"""
    
    @staticmethod
    def trace(func):
        """
        Decorator to trace any function execution
        
        Usage:
        @SimpleTracer.trace
        def my_function():
            pass
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate unique trace ID
            trace_id = str(uuid.uuid4())[:8]
            start = time.time()
            print(f"🚀 [{trace_id}] START {func.__name__}")
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                duration = time.time() - start
                print(f"✅ [{trace_id}] SUCCESS ({duration:.1f}s)")
                SimpleTracer.save_trace(trace_id, "SUCCESS", duration, result)
                return result
                
            except Exception as e:
                duration = time.time() - start
                print(f"❌ [{trace_id}] FAILED ({duration:.1f}s): {e}")
                SimpleTracer.save_trace(trace_id, "FAILED", duration, str(e))
                raise
        
        return wrapper
    
    @staticmethod
    def save_trace(trace_id, status, duration, result):
        """
        Save trace to file for leaderboard metrics
        
        Args:
            trace_id: Unique identifier
            status: SUCCESS/FAILED
            duration: Execution time in seconds
            result: Function result (truncated)
        """
        # Truncate result for storage
        result_str = str(result)
        if len(result_str) > 500:
            result_str = result_str[:500] + "...[TRUNCATED]"
        
        trace = {
            "id": trace_id,
            "status": status,
            "duration": duration,
            "result_preview": result_str,
            "timestamp": time.time()
        }
        
        # Save to file
        with open(f".traces/{trace_id}.json", "w") as f:
            json.dump(trace, f, indent=2)
