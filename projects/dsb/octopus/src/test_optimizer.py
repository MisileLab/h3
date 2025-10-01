
import json
from src.optimizer import optimize_trace

if __name__ == "__main__":
    # Use one of the generated traces for testing
    trace_file = "traces/trace_a049f47e-077b-41b8-82ba-7ff4260d27ca.json"
    
    with open(trace_file, 'r', encoding='utf-8') as f:
        original_trace = json.load(f)
        
    optimized_result = optimize_trace(original_trace)
    
    print("--- Original Trace ---")
    print(json.dumps(original_trace, indent=2, ensure_ascii=False))
    print("\n--- Optimized Trace ---")
    print(json.dumps(optimized_result, indent=2, ensure_ascii=False))
