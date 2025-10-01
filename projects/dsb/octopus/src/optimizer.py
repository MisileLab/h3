
from typing import Dict, Any, List

def is_duplicate_of_previous(turn: Dict[str, Any], previous_turn: Dict[str, Any]) -> bool:
    """Checks if a turn is a duplicate of the previous one based on message content."""
    if not previous_turn:
        return False
    
    current_messages = turn.get('input', {}).get('messages', [])
    previous_messages = previous_turn.get('input', {}).get('messages', [])
    
    if not current_messages or not previous_messages:
        return False

    return current_messages[-1].get('content') == previous_messages[-1].get('content')

def should_remove_turn(turn: Dict[str, Any], previous_turn: Dict[str, Any] = None) -> bool:
    """Applies rule-based filtering to a turn."""
    # Never remove the synthesis node
    if turn['node_name'] == 'synthesis_node':
        return False

    # 1. Error in turn
    if not turn["output"].get("success", True):
        return True
    
    # 2. Empty result (but not for planning)
    if turn['node_name'] != 'planning_node' and not turn["output"].get("result"):
        return True
        
    # 3. Duplicate of previous turn
    if is_duplicate_of_previous(turn, previous_turn):
        return True
        
    # 4. Interrupted turn
    if turn.get("interrupted", False):
        return True
        
    return False

def optimize_trace(original_trace: Dict[str, Any]) -> Dict[str, Any]:
    """Optimizes a trace using rule-based filtering."""
    essential_turns = []
    previous_turn = None
    
    # Keep track of turns to remove
    turns_to_remove = set()

    # First pass: identify turns to remove
    for i, turn in enumerate(original_trace["turns"]):
        # The first turn cannot be a duplicate
        prev = original_trace["turns"][i-1] if i > 0 else None
        if should_remove_turn(turn, prev):
            turns_to_remove.add(turn['turn_id'])

    # Second pass: build the list of essential turns
    for turn in original_trace["turns"]:
        if turn['turn_id'] not in turns_to_remove:
            essential_turns.append(turn)
        
    return {
        "original_trace": original_trace,
        "optimized_turns": essential_turns,
        "reduction_rate": calculate_reduction(original_trace, essential_turns)
    }

def calculate_reduction(original_trace: Dict[str, Any], optimized_turns: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculates the reduction in turns, tokens, and cost."""
    original_turns_count = len(original_trace["turns"])
    optimized_turns_count = len(optimized_turns)
    
    turn_reduction = (original_turns_count - optimized_turns_count) / original_turns_count if original_turns_count > 0 else 0
    
    return {
        "turn_reduction": f"{turn_reduction * 100:.2f}%"
    }
