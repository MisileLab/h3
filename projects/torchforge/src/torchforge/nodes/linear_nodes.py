"""
Linear layer nodes.
"""

from typing import Tuple

from PyQt6.QtGui import QBrush, QColor

from .base_node import BaseNode, BasePropertyWidget


class LinearNode(BaseNode):
  """Linear (Fully Connected) layer node."""
  
  def __init__(self) -> None:
    super().__init__("Linear")
    
    # Default properties
    self.properties = {
      "in_features": 128,
      "out_features": 64,
      "bias": True,
    }
    
    # Set color
    self.setBrush(QBrush(QColor(100, 250, 150)))
    
  def get_layer_type(self) -> str:
    """Return the layer type string."""
    return "Linear"
    
  def get_pytorch_code(self) -> str:
    """Generate PyTorch code for this layer."""
    props = self.properties
    code = f"nn.Linear("
    code += f"in_features={props['in_features']}, "
    code += f"out_features={props['out_features']}"
    
    if not props["bias"]:
      code += f", bias={props['bias']}"
      
    code += ")"
    return code
    
  def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Calculate output shape given input shape."""
    if len(input_shape) < 2:
      raise ValueError("Linear expects at least 2D input (batch, features)")
      
    # For linear layers, we flatten everything except the batch dimension
    batch_size = input_shape[0]
    return (batch_size, self.properties["out_features"])
    
  def create_property_widget(self) -> "BasePropertyWidget":
    """Create property widget for this node."""
    return LinearPropertyWidget(self)
    
  def get_parameter_summary(self) -> str:
    """Get a summary of key parameters."""
    props = self.properties
    return f"{props['in_features']}→{props['out_features']}"
    
  def get_info_text(self) -> str:
    """Get additional info text for the node."""
    return f"Params: {self._calculate_params():,}"
    
  def _calculate_params(self) -> int:
    """Calculate number of parameters."""
    props = self.properties
    params = props["in_features"] * props["out_features"]
    if props["bias"]:
      params += props["out_features"]
    return params


class LinearPropertyWidget(BasePropertyWidget):
  """Property widget for Linear nodes."""
  
  def __init__(self, node: LinearNode) -> None:
    super().__init__(node)
    
  def _add_properties(self) -> None:
    """Add Linear specific properties."""
    self._add_int_property("in_features", self.node.get_property("in_features", 128), 1, 100000)
    self._add_int_property("out_features", self.node.get_property("out_features", 64), 1, 100000)
    
    bias_combo = self._add_combo_property("bias", ["True", "False"], "True")
    bias_combo.currentTextChanged.connect(lambda text: self.value_changed.emit("bias", text == "True"))