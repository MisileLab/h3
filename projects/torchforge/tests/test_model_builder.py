"""
Tests for the model builder.
"""

import pytest
from unittest.mock import Mock

from src.torchforge.core.model_builder import ModelBuilder
from src.torchforge.nodes.conv_nodes import Conv2DNode
from src.torchforge.nodes.linear_nodes import LinearNode
from src.torchforge.nodes.activation_nodes import ReLUNode
from src.torchforge.nodes.utility_nodes import FlattenNode


class TestModelBuilder:
  """Test cases for ModelBuilder."""
  
  def setup_method(self) -> None:
    """Setup test fixtures."""
    self.model_builder = ModelBuilder()
    
  def test_empty_graph_validation(self) -> None:
    """Test validation of empty graph."""
    is_valid, errors = self.model_builder.validate_graph()
    assert not is_valid
    assert "No nodes in the graph" in errors
    
  def test_simple_model_validation(self) -> None:
    """Test validation of a simple model."""
    # Create a simple Conv2D -> ReLU -> Linear model
    conv_node = Conv2DNode()
    conv_node.set_property("in_channels", 3)
    conv_node.set_property("out_channels", 16)
    
    relu_node = ReLUNode()
    flatten_node = FlattenNode()
    linear_node = LinearNode()
    linear_node.set_property("in_features", 16 * 28 * 28)  # Assuming 28x28 input
    linear_node.set_property("out_features", 10)
    
    # Connect nodes
    conv_node.add_output_connection(relu_node)
    relu_node.add_input_connection(conv_node)
    relu_node.add_output_connection(flatten_node)
    flatten_node.add_input_connection(relu_node)
    flatten_node.add_output_connection(linear_node)
    linear_node.add_input_connection(flatten_node)
    
    self.model_builder.set_nodes([conv_node, relu_node, flatten_node, linear_node])
    self.model_builder.set_input_shape((1, 3, 28, 28))
    
    is_valid, errors = self.model_builder.validate_graph()
    assert is_valid, f"Validation failed: {errors}"
    
  def test_code_generation(self) -> None:
    """Test PyTorch code generation."""
    # Create a simple model
    conv_node = Conv2DNode()
    conv_node.set_property("in_channels", 1)
    conv_node.set_property("out_channels", 32)
    conv_node.set_property("kernel_size", 3)
    
    relu_node = ReLUNode()
    flatten_node = FlattenNode()
    linear_node = LinearNode()
    linear_node.set_property("in_features", 32 * 26 * 26)  # 28-3+1 = 26
    linear_node.set_property("out_features", 10)
    
    # Connect nodes
    conv_node.add_output_connection(relu_node)
    relu_node.add_input_connection(conv_node)
    relu_node.add_output_connection(flatten_node)
    flatten_node.add_input_connection(relu_node)
    flatten_node.add_output_connection(linear_node)
    linear_node.add_input_connection(flatten_node)
    
    self.model_builder.set_nodes([conv_node, relu_node, flatten_node, linear_node])
    self.model_builder.set_input_shape((1, 1, 28, 28))
    
    code = self.model_builder.generate_pytorch_code()
    
    # Check that code contains expected elements
    assert "import torch" in code
    assert "class GeneratedModel(nn.Module):" in code
    assert "def __init__(self):" in code
    assert "def forward(self, x):" in code
    assert "nn.Conv2d" in code
    assert "nn.ReLU" in code
    assert "nn.Flatten" in code
    assert "nn.Linear" in code
    
  def test_model_info(self) -> None:
    """Test model information generation."""
    conv_node = Conv2DNode()
    conv_node.set_property("in_channels", 1)
    conv_node.set_property("out_channels", 32)
    conv_node.set_property("kernel_size", 3)
    
    linear_node = LinearNode()
    linear_node.set_property("in_features", 100)
    linear_node.set_property("out_features", 10)
    
    self.model_builder.set_nodes([conv_node, linear_node])
    self.model_builder.set_input_shape((1, 1, 28, 28))
    
    info = self.model_builder.get_model_info()
    
    assert "total_parameters" in info
    assert "num_layers" in info
    assert "layers" in info
    assert info["num_layers"] == 2
    assert len(info["layers"]) == 2
    
  def test_cycle_detection(self) -> None:
    """Test cycle detection in graphs."""
    # Create nodes with a cycle
    node1 = Conv2DNode()
    node2 = ReLUNode()
    
    # Create a cycle: node1 -> node2 -> node1
    node1.add_output_connection(node2)
    node2.add_input_connection(node1)
    node2.add_output_connection(node1)
    node1.add_input_connection(node2)
    
    self.model_builder.set_nodes([node1, node2])
    self.model_builder.set_input_shape((1, 3, 28, 28))
    
    is_valid, errors = self.model_builder.validate_graph()
    assert not is_valid
    assert "contains cycles" in " ".join(errors).lower()
    
  def test_dimension_compatibility(self) -> None:
    """Test dimension compatibility checking."""
    # Create incompatible nodes
    conv_node = Conv2DNode()
    conv_node.set_property("in_channels", 3)
    conv_node.set_property("out_channels", 64)
    
    linear_node = LinearNode()
    linear_node.set_property("in_features", 100)  # Wrong size
    linear_node.set_property("out_features", 10)
    
    # Connect without flatten (should fail)
    conv_node.add_output_connection(linear_node)
    linear_node.add_input_connection(conv_node)
    
    self.model_builder.set_nodes([conv_node, linear_node])
    self.model_builder.set_input_shape((1, 3, 28, 28))
    
    is_valid, errors = self.model_builder.validate_graph()
    assert not is_valid
    # Should have some shape-related error