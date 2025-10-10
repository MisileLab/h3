"""
Activation function nodes.
"""

from typing import Tuple

from PyQt6.QtGui import QBrush, QColor

from .base_node import BaseNode, BasePropertyWidget


class ReLUNode(BaseNode):
  """ReLU activation function node."""
  
  def __init__(self) -> None:
    super().__init__("ReLU")
    
    # Default properties
    self.properties = {
      "inplace": False,
    }
    
    # Set color
    self.setBrush(QBrush(QColor(250, 150, 100)))
    
  def get_layer_type(self) -> str:
    """Return the layer type string."""
    return "ReLU"
    
  def get_pytorch_code(self) -> str:
    """Generate PyTorch code for this layer."""
    props = self.properties
    if props["inplace"]:
      return "nn.ReLU(inplace=True)"
    else:
      return "nn.ReLU()"
    
  def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Calculate output shape given input shape."""
    # ReLU doesn't change shape
    return input_shape
    
  def create_property_widget(self) -> "BasePropertyWidget":
    """Create property widget for this node."""
    return ReLUPropertyWidget(self)
    
  def get_parameter_summary(self) -> str:
    """Get a summary of key parameters."""
    return "max(0, x)"
    
  def get_info_text(self) -> str:
    """Get additional info text for the node."""
    return "No parameters"


class ReLUPropertyWidget(BasePropertyWidget):
  """Property widget for ReLU nodes."""
  
  def __init__(self, node: ReLUNode) -> None:
    super().__init__(node)
    
  def _add_properties(self) -> None:
    """Add ReLU specific properties."""
    inplace_combo = self._add_combo_property("inplace", ["False", "True"], "False")
    inplace_combo.currentTextChanged.connect(lambda text: self.value_changed.emit("inplace", text == "True"))


class SigmoidNode(BaseNode):
  """Sigmoid activation function node."""
  
  def __init__(self) -> None:
    super().__init__("Sigmoid")
    
    # Default properties
    self.properties = {}
    
    # Set color
    self.setBrush(QBrush(QColor(250, 100, 150)))
    
  def get_layer_type(self) -> str:
    """Return the layer type string."""
    return "Sigmoid"
    
  def get_pytorch_code(self) -> str:
    """Generate PyTorch code for this layer."""
    return "nn.Sigmoid()"
    
  def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Calculate output shape given input shape."""
    # Sigmoid doesn't change shape
    return input_shape
    
  def create_property_widget(self) -> "BasePropertyWidget":
    """Create property widget for this node."""
    return SigmoidPropertyWidget(self)
    
  def get_parameter_summary(self) -> str:
    """Get a summary of key parameters."""
    return "1/(1+e^-x)"
    
  def get_info_text(self) -> str:
    """Get additional info text for the node."""
    return "No parameters"


class SigmoidPropertyWidget(BasePropertyWidget):
  """Property widget for Sigmoid nodes."""
  
  def __init__(self, node: SigmoidNode) -> None:
    super().__init__(node)
    
  def _add_properties(self) -> None:
    """Add Sigmoid specific properties."""
    # Sigmoid has no configurable properties
    pass