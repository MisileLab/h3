"""
Graph canvas for visual model building.
"""

from PyQt6.QtWidgets import (
  QGraphicsView,
  QGraphicsScene,
  QGraphicsRectItem,
  QVBoxLayout,
  QWidget,
  QScrollArea,
)
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPen, QBrush, QColor, QPainter, QFont

from .node import BaseNode
from .edge import Edge
from .toolbar import GraphToolbar


class GraphCanvas(QGraphicsView):
  """Main canvas for building neural network graphs."""
  
  def __init__(self) -> None:
    super().__init__()
    self._setup_scene()
    self._setup_ui()
    self._setup_interactions()
    
    # Store nodes and edges
    self.nodes: list[BaseNode] = []
    self.edges: list[Edge] = []
    
    # Connection state
    self._connection_start: BaseNode | None = None
    self._temp_edge: Edge | None = None
    
  def _setup_scene(self) -> None:
    """Setup the graphics scene."""
    self.scene = QGraphicsScene()
    self.scene.setSceneRect(-2000, -2000, 4000, 4000)
    self.setScene(self.scene)
    
    # Set background
    self.setBackgroundBrush(QBrush(QColor(240, 240, 240)))
    
    # Add grid
    self._add_grid()
    
  def _add_grid(self) -> None:
    """Add background grid to the scene."""
    grid_size = 20
    pen = QPen(QColor(200, 200, 200), 0.5)
    
    # Vertical lines
    for x in range(-2000, 2001, grid_size):
      self.scene.addLine(x, -2000, x, 2000, pen)
    
    # Horizontal lines
    for y in range(-2000, 2001, grid_size):
      self.scene.addLine(-2000, y, 2000, y, pen)
      
  def _setup_ui(self) -> None:
    """Setup the UI components."""
    # Enable antialiasing
    self.setRenderHint(QPainter.RenderHint.Antialiasing)
    self.setRenderHint(QPainter.RenderHint.TextAntialiasing)
    
    # Set drag mode
    self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
    
    # Set transformation anchor
    self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
    self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
    
  def _setup_interactions(self) -> None:
    """Setup user interactions."""
    # Enable mouse tracking
    self.setMouseTracking(True)
    
  def add_node(self, node: BaseNode, position: QPointF | None = None) -> None:
    """Add a node to the canvas."""
    if position is None:
      position = QPointF(0, 0)
    
    node.setPos(position)
    self.scene.addItem(node)
    self.nodes.append(node)
    
    # Connect node signals
    node.node_clicked.connect(self._on_node_clicked)
    node.connection_started.connect(self._on_connection_started)
    node.connection_finished.connect(self._on_connection_finished)
    
  def remove_node(self, node: BaseNode) -> None:
    """Remove a node from the canvas."""
    if node in self.nodes:
      # Remove connected edges
      edges_to_remove = [edge for edge in self.edges if edge.source == node or edge.target == node]
      for edge in edges_to_remove:
        self.remove_edge(edge)
      
      # Remove node
      self.scene.removeItem(node)
      self.nodes.remove(node)
      
  def add_edge(self, edge: Edge) -> None:
    """Add an edge to the canvas."""
    self.scene.addItem(edge)
    self.edges.append(edge)
    
  def remove_edge(self, edge: Edge) -> None:
    """Remove an edge from the canvas."""
    if edge in self.edges:
      self.scene.removeItem(edge)
      self.edges.remove(edge)
      
  def _on_node_clicked(self, node: BaseNode) -> None:
    """Handle node click events."""
    # Emit signal for property panel update
    pass
    
  def _on_connection_started(self, node: BaseNode) -> None:
    """Handle connection start from a node."""
    self._connection_start = node
    
    # Create temporary edge
    self._temp_edge = Edge(node, None)
    self.add_edge(self._temp_edge)
    
  def _on_connection_finished(self, node: BaseNode) -> None:
    """Handle connection finish at a node."""
    if self._connection_start and self._connection_start != node:
      # Create permanent edge
      edge = Edge(self._connection_start, node)
      self.add_edge(edge)
      
      # Connect nodes
      self._connection_start.add_output_connection(node)
      node.add_input_connection(self._connection_start)
    
    # Clean up temporary edge
    if self._temp_edge:
      self.remove_edge(self._temp_edge)
      self._temp_edge = None
      
    self._connection_start = None
    
  def wheelEvent(self, event) -> None:
    """Handle mouse wheel for zooming."""
    zoom_in_factor = 1.25
    zoom_out_factor = 1 / zoom_in_factor
    
    # Save the scene pos
    old_pos = self.mapToScene(event.position().toPoint())
    
    # Zoom
    if event.angleDelta().y() > 0:
      zoom_factor = zoom_in_factor
    else:
      zoom_factor = zoom_out_factor
      
    self.scale(zoom_factor, zoom_factor)
    
    # Get the new position
    new_pos = self.mapToScene(event.position().toPoint())
    
    # Move scene to old position
    delta = new_pos - old_pos
    self.translate(delta.x(), delta.y())
    
  def mousePressEvent(self, event) -> None:
    """Handle mouse press events."""
    if event.button() == Qt.MouseButton.MiddleButton:
      # Start panning
      self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
      event.accept()
      return
      
    super().mousePressEvent(event)
    
  def mouseReleaseEvent(self, event) -> None:
    """Handle mouse release events."""
    if event.button() == Qt.MouseButton.MiddleButton:
      # Stop panning
      self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
      event.accept()
      return
      
    super().mouseReleaseEvent(event)
    
  def clear_graph(self) -> None:
    """Clear all nodes and edges from the canvas."""
    # Remove all edges
    for edge in self.edges.copy():
      self.remove_edge(edge)
      
    # Remove all nodes
    for node in self.nodes.copy():
      self.remove_node(node)
      
  def validate_graph(self) -> tuple[bool, list[str]]:
    """Validate the graph structure."""
    errors: list[str] = []
    
    # Check for empty graph
    if not self.nodes:
      errors.append("Graph is empty")
      return False, errors
      
    # Check for disconnected nodes
    input_nodes = [node for node in self.nodes if not node.input_connections]
    output_nodes = [node for node in self.nodes if not node.output_connections]
    
    if len(input_nodes) != 1:
      errors.append(f"Graph should have exactly 1 input node, found {len(input_nodes)}")
      
    if len(output_nodes) != 1:
      errors.append(f"Graph should have exactly 1 output node, found {len(output_nodes)}")
      
    # Check for cycles
    if self._has_cycles():
      errors.append("Graph contains cycles")
      
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