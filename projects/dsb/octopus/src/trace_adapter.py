from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict, List, Union
from langchain.schema import AgentAction, AgentFinish, LLMResult

class LangChainTraceAdapter(BaseCallbackHandler):
    """
    A callback handler to adapt LangChain agent traces to the format expected by our optimizer.
    """
    def __init__(self):
        self.turns: List[Dict[str, Any]] = []
        self._turn_id_counter = 1
        self._last_action: AgentAction = None

    def _create_turn(self, node_name: str, input_content: Any, output_content: Any = None, success: bool = True) -> Dict[str, Any]:
        """Creates a new turn and adds it to the list."""
        turn = {
            "turn_id": self._turn_id_counter,
            "node_name": node_name,
            "input": {"messages": [{"content": str(input_content)}]},
            "output": {"success": success, "result": str(output_content) if output_content else ""},
            "interrupted": False
        }
        self.turns.append(turn)
        self._turn_id_counter += 1
        return turn

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Capture the start of a tool action."""
        self._last_action = action
        # We create the turn here, but the result will be filled in on_tool_end
        self._create_turn(
            node_name=f"tool: {action.tool}",
            input_content=action.tool_input
        )

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Update the last turn with the tool's output."""
        if self.turns:
            # The last created turn should be the one from on_agent_action
            self.turns[-1]["output"]["result"] = output

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> Any:
        """Mark the last turn as failed on tool error."""
        if self.turns:
            self.turns[-1]["output"]["success"] = False
            self.turns[-1]["output"]["result"] = str(error)

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Capture the final synthesis step of the agent."""
        self._create_turn(
            node_name="synthesis_node",
            input_content=finish.return_values,
            output_content=finish.log
        )

    def get_trace(self) -> Dict[str, List[Dict]]:
        """Returns the captured trace in the optimizer's expected format."""
        return {"turns": self.turns}
