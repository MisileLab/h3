"""
Utility layer nodes.
"""

from typing import Tuple

from PyQt6.QtGui import QBrush, QColor

from .base_node import BaseNode, BasePropertyWidget


class FlattenNode(BaseNode):
  """Flatten layer node."""
  
  def __init__(self) -> None:
    super().__init__("Flatten")
    
    # Default properties
    self.properties = {
      "start_dim": 1,
      "end_dim": -1,
    }
    
    # Set color
    self.setBrush(QBrush(QColor(200, 200, 200)))
    
  def get_layer_type(self) -> str:
    """Return the layer type string."""
    return "Flatten"
    
  def get_pytorch_code(self) -> str:
    """Generate PyTorch code for this layer."""
    props = self.properties
    if props["start_dim"] == 1 and props["end_dim"] == -1:
      return "nn.Flatten()"
    else:
      return f"nn.Flatten(start_dim={props['start_dim']}, end_dim={props['end_dim']})"
    
  def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Calculate output shape given input shape."""
    props = self.properties
    
    # Handle negative end_dim
    end_dim = props["end_dim"] if props["end_dim"] >= 0 else len(input_shape) + props["end_dim"]
    
    # Calculate flattened dimensions
    before_dims = input_shape[:props["start_dim"]]
    flattened_dim = 1
    for i in range(props["start_dim"], end_dim + 1):
      flattened_dim *= input_shape[i]
    after_dims = input_shape[end_dim + 1:] if end_dim + 1 < len(input_shape) else ()
    
    return before_dims + (flattened_dim,) + after_dims
    
  def create_property_widget(self) -> "BasePropertyWidget":
    """Create property widget for this node."""
    return FlattenPropertyWidget(self)
    
  def get_parameter_summary(self) -> str:
    """Get a summary of key parameters."""
    props = self.properties
    if props["start_dim"] == 1 and props["end_dim"] == -1:
      return "1D"
    else:
      return f"{props['start_dim']}→{props['end_dim']}"
    
  def get_info_text(self) -> str:
    """Get additional info text for the node."""
    return "No parameters"


class FlattenPropertyWidget(BasePropertyWidget):
  """Property widget for Flatten nodes."""
  
  def __init__(self, node: FlattenNode) -> None:
    super().__init__(node)
    
  def _add_properties(self) -> None:
    """Add Flatten specific properties."""
    self._add_int_property("start_dim", self.node.get_property("start_dim", 1), 0, 10)
    self._add_int_property("end_dim", self.node.get_property("end_dim", -1), -10, 10)