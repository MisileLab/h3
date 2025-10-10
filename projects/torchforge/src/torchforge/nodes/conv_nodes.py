"""
Convolutional layer nodes.
"""

from typing import Tuple, Any

from PyQt6.QtWidgets import QWidget, QGroupBox, QFormLayout, QComboBox
from PyQt6.QtGui import QBrush, QColor

from .base_node import BaseNode, BasePropertyWidget


class Conv2DNode(BaseNode):
  """2D Convolution layer node."""
  
  def __init__(self) -> None:
    super().__init__("Conv2D")
    
    # Default properties
    self.properties = {
      "in_channels": 3,
      "out_channels": 32,
      "kernel_size": 3,
      "stride": 1,
      "padding": 0,
      "dilation": 1,
      "groups": 1,
      "bias": True,
    }
    
    # Set color
    self.setBrush(QBrush(QColor(100, 150, 250)))
    
  def get_layer_type(self) -> str:
    """Return the layer type string."""
    return "Conv2D"
    
  def get_pytorch_code(self) -> str:
    """Generate PyTorch code for this layer."""
    props = self.properties
    code = f"nn.Conv2d("
    code += f"in_channels={props['in_channels']}, "
    code += f"out_channels={props['out_channels']}, "
    code += f"kernel_size={props['kernel_size']}, "
    
    if props["stride"] != 1:
      code += f"stride={props['stride']}, "
    if props["padding"] != 0:
      code += f"padding={props['padding']}, "
    if props["dilation"] != 1:
      code += f"dilation={props['dilation']}, "
    if props["groups"] != 1:
      code += f"groups={props['groups']}, "
    if not props["bias"]:
      code += f"bias={props['bias']}, "
      
    code = code.rstrip(", ") + ")"
    return code
    
  def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Calculate output shape given input shape."""
    if len(input_shape) != 4:  # (batch, channels, height, width)
      raise ValueError("Conv2D expects 4D input (batch, channels, height, width)")
      
    batch, _, h, w = input_shape
    props = self.properties
    
    # Calculate output height and width
    h_out = ((h + 2 * props["padding"] - props["dilation"] * (props["kernel_size"] - 1) - 1) // props["stride"]) + 1
    w_out = ((w + 2 * props["padding"] - props["dilation"] * (props["kernel_size"] - 1) - 1) // props["stride"]) + 1
    
    return (batch, props["out_channels"], h_out, w_out)
    
  def create_property_widget(self) -> QWidget:
    """Create property widget for this node."""
    return Conv2DPropertyWidget(self)
    
  def get_parameter_summary(self) -> str:
    """Get a summary of key parameters."""
    props = self.properties
    return f"{props['in_channels']}→{props['out_channels']}, k{props['kernel_size']}"
    
  def get_info_text(self) -> str:
    """Get additional info text for the node."""
    props = self.properties
    return f"Params: {self._calculate_params():,}"
    
  def _calculate_params(self) -> int:
    """Calculate number of parameters."""
    props = self.properties
    params = props["out_channels"] * props["in_channels"] * props["kernel_size"] * props["kernel_size"]
    if props["bias"]:
      params += props["out_channels"]
    return params


class Conv2DPropertyWidget(BasePropertyWidget):
  """Property widget for Conv2D nodes."""
  
  def __init__(self, node: Conv2DNode) -> None:
    super().__init__(node)
    
  def _add_properties(self) -> None:
    """Add Conv2D specific properties."""
    self._add_int_property("in_channels", self.node.get_property("in_channels", 3), 1, 1024)
    self._add_int_property("out_channels", self.node.get_property("out_channels", 32), 1, 1024)
    self._add_int_property("kernel_size", self.node.get_property("kernel_size", 3), 1, 11)
    self._add_int_property("stride", self.node.get_property("stride", 1), 1, 10)
    self._add_int_property("padding", self.node.get_property("padding", 0), 0, 10)
    self._add_int_property("dilation", self.node.get_property("dilation", 1), 1, 10)
    self._add_int_property("groups", self.node.get_property("groups", 1), 1, 1024)
    
    bias_combo = self._add_combo_property("bias", ["True", "False"], "True")
    bias_combo.currentTextChanged.connect(lambda text: self.value_changed.emit("bias", text == "True"))


class MaxPool2DNode(BaseNode):
  """2D Max Pooling layer node."""
  
  def __init__(self) -> None:
    super().__init__("MaxPool2D")
    
    # Default properties
    self.properties = {
      "kernel_size": 2,
      "stride": 2,
      "padding": 0,
      "dilation": 1,
      "return_indices": False,
      "ceil_mode": False,
    }
    
    # Set color
    self.setBrush(QBrush(QColor(150, 100, 250)))
    
  def get_layer_type(self) -> str:
    """Return the layer type string."""
    return "MaxPool2D"
    
  def get_pytorch_code(self) -> str:
    """Generate PyTorch code for this layer."""
    props = self.properties
    code = f"nn.MaxPool2d("
    code += f"kernel_size={props['kernel_size']}, "
    
    if props["stride"] != props["kernel_size"]:
      code += f"stride={props['stride']}, "
    if props["padding"] != 0:
      code += f"padding={props['padding']}, "
    if props["dilation"] != 1:
      code += f"dilation={props['dilation']}, "
    if props["return_indices"]:
      code += f"return_indices={props['return_indices']}, "
    if props["ceil_mode"]:
      code += f"ceil_mode={props['ceil_mode']}, "
      
    code = code.rstrip(", ") + ")"
    return code
    
  def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Calculate output shape given input shape."""
    if len(input_shape) != 4:  # (batch, channels, height, width)
      raise ValueError("MaxPool2D expects 4D input (batch, channels, height, width)")
      
    batch, channels, h, w = input_shape
    props = self.properties
    
    # Calculate output height and width
    if props["ceil_mode"]:
      h_out = ((h + 2 * props["padding"] - props["dilation"] * (props["kernel_size"] - 1) - 1 + props["stride"] - 1) // props["stride"]) + 1
      w_out = ((w + 2 * props["padding"] - props["dilation"] * (props["kernel_size"] - 1) - 1 + props["stride"] - 1) // props["stride"]) + 1
    else:
      h_out = ((h + 2 * props["padding"] - props["dilation"] * (props["kernel_size"] - 1) - 1) // props["stride"]) + 1
      w_out = ((w + 2 * props["padding"] - props["dilation"] * (props["kernel_size"] - 1) - 1) // props["stride"]) + 1
    
    return (batch, channels, h_out, w_out)
    
  def create_property_widget(self) -> QWidget:
    """Create property widget for this node."""
    return MaxPool2DPropertyWidget(self)
    
  def get_parameter_summary(self) -> str:
    """Get a summary of key parameters."""
    props = self.properties
    return f"k{props['kernel_size']}, s{props['stride']}"
    
  def get_info_text(self) -> str:
    """Get additional info text for the node."""
    return "No parameters"


class MaxPool2DPropertyWidget(BasePropertyWidget):
  """Property widget for MaxPool2D nodes."""
  
  def __init__(self, node: MaxPool2DNode) -> None:
    super().__init__(node)
    
  def _add_properties(self) -> None:
    """Add MaxPool2D specific properties."""
    self._add_int_property("kernel_size", self.node.get_property("kernel_size", 2), 1, 11)
    self._add_int_property("stride", self.node.get_property("stride", 2), 1, 10)
    self._add_int_property("padding", self.node.get_property("padding", 0), 0, 10)
    self._add_int_property("dilation", self.node.get_property("dilation", 1), 1, 10)
    
    return_indices_combo = self._add_combo_property("return_indices", ["False", "True"], "False")
    return_indices_combo.currentTextChanged.connect(lambda text: self.value_changed.emit("return_indices", text == "True"))
    
    ceil_mode_combo = self._add_combo_property("ceil_mode", ["False", "True"], "False")
    ceil_mode_combo.currentTextChanged.connect(lambda text: self.value_changed.emit("ceil_mode", text == "True"))