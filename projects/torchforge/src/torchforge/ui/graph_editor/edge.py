"""
Edge class for connecting nodes in the graph.
"""

from typing import Optional

from PyQt6.QtWidgets import QGraphicsItem
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal, QObject
from PyQt6.QtGui import (
  QPen,
  QBrush,
  QColor,
  QPainter,
  QPainterPath,
)


class EdgeSignals(QObject):
  """Signal emitter for edge events."""
  edge_clicked = pyqtSignal(object)
  connection_changed = pyqtSignal(object, object, object)


class Edge(QGraphicsItem):
  """Edge connecting two nodes in the graph."""
  
  def __init__(self, source: "BaseNode", target: Optional["BaseNode"] = None) -> None:
    super().__init__()
    
    self.source = source
    self.target = target
    self.signals = EdgeSignals()
    
    # Appearance
    self._pen = QPen(QColor(50, 50, 50), 3)
    self._pen.setStyle(Qt.PenStyle.SolidLine)
    
    # Arrow properties
    self._arrow_size = 10
    
    # Make edge selectable
    self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
    self.setZValue(-1)  # Draw behind nodes
    
    # Set initial position
    if source:
      self.update_position()
      
  def set_target(self, target: "BaseNode") -> None:
    """Set the target node."""
    self.target = target
    self.update_position()
    self.signals.connection_changed.emit(self, self.source, self.target)
    
  def update_position(self) -> None:
    """Update edge position based on node positions."""
    self.prepareGeometryChange()
    if self.source:
      self.update()
      
  def boundingRect(self) -> QRectF:
    """Return the bounding rectangle of the edge."""
    if not self.source:
      return QRectF()
      
    start_pos = self.source.get_output_port_pos()
    
    if self.target:
      end_pos = self.target.get_input_port_pos()
    else:
      # Temporary edge - use current mouse position
      end_pos = start_pos
      
    # Calculate bounding rectangle with padding
    padding = self._arrow_size + 5
    x_min = min(start_pos.x(), end_pos.x()) - padding
    y_min = min(start_pos.y(), end_pos.y()) - padding
    x_max = max(start_pos.x(), end_pos.x()) + padding
    y_max = max(start_pos.y(), end_pos.y()) + padding
    
    return QRectF(x_min, y_min, x_max - x_min, y_max - y_min)
    
  def paint(self, painter: QPainter, option: any, widget: any) -> None:
    """Paint the edge."""
    if not self.source:
      return
      
    # Set pen
    if self.isSelected():
      pen = QPen(QColor(100, 150, 200), 4)
    else:
      pen = self._pen
    painter.setPen(pen)
    
    # Get start and end positions
    start_pos = self.source.get_output_port_pos()
    
    if self.target:
      end_pos = self.target.get_input_port_pos()
    else:
      # For temporary edges, draw to current scene position
      # This would be updated from mouse move events
      end_pos = start_pos + QPointF(100, 0)
      
    # Create curved path
    path = self._create_path(start_pos, end_pos)
    
    # Draw the path
    painter.drawPath(path)
    
    # Draw arrow at the end
    if self.target:
      self._draw_arrow(painter, path)
      
  def _create_path(self, start: QPointF, end: QPointF) -> QPainterPath:
    """Create a curved path from start to end."""
    path = QPainterPath()
    
    # Calculate control points for a smooth curve
    dx = end.x() - start.x()
    dy = end.y() - start.y()
    
    # Use cubic Bezier curve for smooth connection
    ctrl1 = QPointF(start.x() + dx * 0.5, start.y())
    ctrl2 = QPointF(end.x() - dx * 0.5, end.y())
    
    path.moveTo(start)
    path.cubicTo(ctrl1, ctrl2, end)
    
    return path
    
  def _draw_arrow(self, painter: QPainter, path: QPainterPath) -> None:
    """Draw arrow at the end of the path."""
    # Get the angle at the end of the path
    percent = path.percentAtLength(path.length() - 1)
    end_point = path.pointAtPercent(percent)
    
    # Calculate angle
    delta = 0.01
    prev_point = path.pointAtPercent(max(0, percent - delta))
    angle = atan2(end_point.y() - prev_point.y(), end_point.x() - prev_point.x())
    
    # Create arrow polygon
    arrow_path = QPainterPath()
    arrow_path.moveTo(end_point)
    arrow_path.lineTo(
      end_point.x() - self._arrow_size * cos(angle + 2.5),
      end_point.y() - self._arrow_size * sin(angle + 2.5)
    )
    arrow_path.lineTo(
      end_point.x() - self._arrow_size * cos(angle - 2.5),
      end_point.y() - self._arrow_size * sin(angle - 2.5)
    )
    arrow_path.closeSubpath()
    
    # Draw arrow
    painter.setBrush(QBrush(self._pen.color()))
    painter.drawPath(arrow_path)
    
  def mousePressEvent(self, event) -> None:
    """Handle mouse press events."""
    if event.button() == Qt.MouseButton.LeftButton:
      self.signals.edge_clicked.emit(self)
      event.accept()
      return
      
    super().mousePressEvent(event)
    
  def to_dict(self) -> dict:
    """Convert edge to dictionary for serialization."""
    return {
      "source": self.source.node_id if self.source else None,
      "target": self.target.node_id if self.target else None,
    }


# Helper functions for arrow drawing
import math


def atan2(y: float, x: float) -> float:
  """Calculate arctangent2."""
  return math.atan2(y, x)


def cos(x: float) -> float:
  """Calculate cosine."""
  return math.cos(x)


def sin(x: float) -> float:
  """Calculate sine."""
  return math.sin(x)