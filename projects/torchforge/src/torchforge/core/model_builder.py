"""
Model builder for converting graphs to PyTorch code.
"""

from typing import List, Dict, Any, Tuple, Optional
import textwrap

from ..ui.graph_editor.node import BaseNode


class ModelBuilder:
  """Builds PyTorch models from graph nodes."""
  
  def __init__(self) -> None:
    self.nodes: List[BaseNode] = []
    self.input_shape: Optional[Tuple[int, ...]] = None
    
  def set_nodes(self, nodes: List[BaseNode]) -> None:
    """Set the nodes to build the model from."""
    self.nodes = nodes
    
  def set_input_shape(self, input_shape: Tuple[int, ...]) -> None:
    """Set the input shape for the model."""
    self.input_shape = input_shape
    
  def validate_graph(self) -> Tuple[bool, List[str]]:
    """Validate the graph structure."""
    errors: List[str] = []
    
    if not self.nodes:
      errors.append("No nodes in the graph")
      return False, errors
      
    if self.input_shape is None:
      errors.append("Input shape not set")
      return False, errors
      
    # Find input and output nodes
    input_nodes = [node for node in self.nodes if not node.input_connections]
    output_nodes = [node for node in self.nodes if not node.output_connections]
    
    if len(input_nodes) != 1:
      errors.append(f"Expected exactly 1 input node, found {len(input_nodes)}")
      
    if len(output_nodes) != 1:
      errors.append(f"Expected exactly 1 output node, found {len(output_nodes)}")
      
    # Check for cycles
    if self._has_cycles():
      errors.append("Graph contains cycles")
      
    # Check dimension compatibility
    try:
      self._validate_dimensions()
    except ValueError as e:
      errors.append(str(e))
      
    return len(errors) == 0, errors
    
  def _has_cycles(self) -> bool:
    """Check if the graph contains cycles using DFS."""
    visited = set()
    rec_stack = set()
    
    def visit(node: BaseNode) -> bool:
      if node in rec_stack:
        return True
      if node in visited:
        return False
        
      visited.add(node)
      rec_stack.add(node)
      
      for output_node in node.output_connections:
        if visit(output_node):
          return True
          
      rec_stack.remove(node)
      return False
      
    for node in self.nodes:
      if node not in visited:
        if visit(node):
          return True
          
    return False
    
  def _validate_dimensions(self) -> None:
    """Validate dimension compatibility between nodes."""
    if self.input_shape is None:
      raise ValueError("Input shape not set")
      
    # Find input node
    input_nodes = [node for node in self.nodes if not node.input_connections]
    if not input_nodes:
      raise ValueError("No input node found")
      
    current_shape = self.input_shape
    visited = set()
    
    def propagate_shape(node: BaseNode, shape: Tuple[int, ...]) -> Tuple[int, ...]:
      """Propagate shape through the graph."""
      if node in visited:
        return shape
        
      visited.add(node)
      
      try:
        output_shape = node.get_output_shape(shape)
      except Exception as e:
        raise ValueError(f"Shape error at {node.title}: {e}")
        
      # Propagate to output nodes
      for output_node in node.output_connections:
        propagate_shape(output_node, output_shape)
        
      return output_shape
      
    # Start from input node
    input_node = input_nodes[0]
    propagate_shape(input_node, current_shape)
    
  def generate_pytorch_code(self, class_name: str = "GeneratedModel") -> str:
    """Generate PyTorch code from the graph."""
    if not self.nodes:
      raise ValueError("No nodes to generate code from")
      
    # Validate first
    is_valid, errors = self.validate_graph()
    if not is_valid:
      raise ValueError(f"Invalid graph: {'; '.join(errors)}")
      
    # Find input node and topological order
    input_nodes = [node for node in self.nodes if not node.input_connections]
    if not input_nodes:
      raise ValueError("No input node found")
      
    input_node = input_nodes[0]
    ordered_nodes = self._topological_sort(input_node)
    
    # Generate imports
    imports = [
      "import torch",
      "import torch.nn as nn",
      "import torch.nn.functional as F",
    ]
    
    # Generate class definition
    class_def = f"\n\nclass {class_name}(nn.Module):"
    
    # Generate __init__ method
    init_lines = [
      "    def __init__(self):",
      "        super().__init__()",
    ]
    
    # Add layer definitions
    layer_names = []
    for i, node in enumerate(ordered_nodes):
      layer_name = f"layer_{i}_{node.title.lower().replace(' ', '_')}"
      layer_names.append(layer_name)
      
      # Generate layer code
      layer_code = node.get_pytorch_code()
      init_lines.append(f"        self.{layer_name} = {layer_code}")
      
    # Generate forward method
    forward_lines = [
      "",
      "    def forward(self, x):",
    ]
    
    # Add forward pass
    for i, (node, layer_name) in enumerate(zip(ordered_nodes, layer_names)):
      if i == 0:
        forward_lines.append(f"        x = self.{layer_name}(x)")
      else:
        forward_lines.append(f"        x = self.{layer_name}(x)")
        
    forward_lines.append("        return x")
    
    # Combine all parts
    code_parts = imports + [class_def] + init_lines + forward_lines
    
    # Add example usage
    example = [
      "",
      "",
      "# Example usage:",
      f"model = {class_name}()",
      "print(model)",
      "",
      "# Test with dummy input",
      f"dummy_input = torch.randn(1, {', '.join(map(str, self.input_shape[1:]))})",
      "output = model(dummy_input)",
      "print(f'Output shape: {output.shape}')",
    ]
    
    code_parts.extend(example)
    
    return "\n".join(code_parts)
    
  def _topological_sort(self, start_node: BaseNode) -> List[BaseNode]:
    """Perform topological sort starting from the input node."""
    visited = set()
    result: List[BaseNode] = []
    
    def dfs(node: BaseNode) -> None:
      if node in visited:
        return
        
      visited.add(node)
      
      # Visit output nodes first
      for output_node in node.output_connections:
        dfs(output_node)
        
      result.append(node)
      
    dfs(start_node)
    return result[::-1]  # Reverse to get input-to-output order
    
  def get_model_info(self) -> Dict[str, Any]:
    """Get information about the generated model."""
    if not self.nodes:
      return {}
      
    total_params = 0
    layer_info = []
    
    for node in self.nodes:
      # Calculate parameters for this node
      if hasattr(node, '_calculate_params'):
        params = node._calculate_params()
      else:
        params = 0  # For layers without parameters
        
      total_params += params
      
      layer_info.append({
        "name": node.title,
        "type": node.get_layer_type(),
        "parameters": params,
        "properties": node.properties.copy(),
      })
      
    return {
      "total_parameters": total_params,
      "num_layers": len(self.nodes),
      "layers": layer_info,
      "input_shape": self.input_shape,
    }