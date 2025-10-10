"""
Base node implementation for all layer types.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import uuid

from PyQt6.QtWidgets import (
  QGraphicsItem,
  QGraphicsRectItem,
  QGraphicsTextItem,
  QWidget,
  QVBoxLayout,
  QHBoxLayout,
  QLabel,
  QSpinBox,
  QDoubleSpinBox,
  QComboBox,
  QLineEdit,
  QGroupBox,
  QFormLayout,
)
from PyQt6.QtCore import (
  Qt,
  QRectF,
  QPointF,
  pyqtSignal,
  QObject,
)
from PyQt6.QtGui import (
  QPen,
  QBrush,
  QColor,
  QPainter,
  QPainterPath,
  QFont,
)


class NodeSignals(QObject):
  """Signal emitter for node events."""
  node_clicked = pyqtSignal(object)
  connection_started = pyqtSignal(object)
  connection_finished = pyqtSignal(object)
  property_changed = pyqtSignal(str, object)


class BaseNode(QGraphicsRectItem, ABC):
  """Base class for all nodes in the graph."""
  
  def __init__(self, title: str) -> None:
    super().__init__(0, 0, 200, 120)
    
    self.node_id = str(uuid.uuid4())[:8]
    self.title = title
    self.signals = NodeSignals()
    
    # Node properties
    self.properties: Dict[str, Any] = {}
    
    # Connections
    self.input_connections: List["BaseNode"] = []
    self.output_connections: List["BaseNode"] = []
    
    # UI elements
    self._setup_ui()
    self._setup_ports()
    
    # Appearance
    self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
    self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
    self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
    
    # Default colors
    self._default_brush = QBrush(QColor(100, 150, 200))
    self._selected_brush = QBrush(QColor(120, 170, 220))
    self.setBrush(self._default_brush)
    
    self._pen = QPen(QColor(50, 50, 50), 2)
    self.setPen(self._pen)
    
  def _setup_ui(self) -> None:
    """Setup the node UI."""
    # Title text
    self.title_item = QGraphicsTextItem(self.title, self)
    self.title_item.setPos(10, 10)
    self.title_item.setDefaultTextColor(Qt.GlobalColor.white)
    
    font = QFont()
    font.setBold(True)
    font.setPointSize(10)
    self.title_item.setFont(font)
    
    # Subtitle (layer type)
    self.subtitle_item = QGraphicsTextItem(self.get_layer_type(), self)
    self.subtitle_item.setPos(10, 30)
    self.subtitle_item.setDefaultTextColor(Qt.GlobalColor.white)
    
    font.setBold(False)
    font.setPointSize(8)
    self.subtitle_item.setFont(font)
    
  def _setup_ports(self) -> None:
    """Setup input and output ports."""
    # Input port (left side)
    self.input_port = QGraphicsRectItem(-10, 50, 10, 20, self)
    self.input_port.setBrush(QBrush(QColor(200, 100, 100)))
    self.input_port.setPen(QPen(Qt.GlobalColor.black, 1))
    
    # Output port (right side)
    self.output_port = QGraphicsRectItem(200, 50, 10, 20, self)
    self.output_port.setBrush(QBrush(QColor(100, 200, 100)))
    self.output_port.setPen(QPen(Qt.GlobalColor.black, 1))
    
  @abstractmethod
  def get_layer_type(self) -> str:
    """Return the layer type string."""
    pass
    
  @abstractmethod
  def get_pytorch_code(self) -> str:
    """Generate PyTorch code for this layer."""
    pass
    
  @abstractmethod
  def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Calculate output shape given input shape."""
    pass
    
  @abstractmethod
  def create_property_widget(self) -> QWidget:
    """Create property widget for this node."""
    pass
    
  def add_input_connection(self, node: "BaseNode") -> None:
    """Add an input connection."""
    if node not in self.input_connections:
      self.input_connections.append(node)
      
  def add_output_connection(self, node: "BaseNode") -> None:
    """Add an output connection."""
    if node not in self.output_connections:
      self.output_connections.append(node)
      
  def remove_input_connection(self, node: "BaseNode") -> None:
    """Remove an input connection."""
    if node in self.input_connections:
      self.input_connections.remove(node)
      
  def remove_output_connection(self, node: "BaseNode") -> None:
    """Remove an output connection."""
    if node in self.output_connections:
      self.output_connections.remove(node)
      
  def set_property(self, name: str, value: Any) -> None:
    """Set a property value."""
    self.properties[name] = value
    self.signals.property_changed.emit(name, value)
    self.update_display()
    
  def get_property(self, name: str, default: Any = None) -> Any:
    """Get a property value."""
    return self.properties.get(name, default)
    
  def update_display(self) -> None:
    """Update the node display based on properties."""
    # Update subtitle with key parameters
    param_text = self.get_parameter_summary()
    self.subtitle_item.setPlainText(f"{self.get_layer_type()}: {param_text}")
    
  def get_parameter_summary(self) -> str:
    """Get a summary of key parameters."""
    return "default"
    
  def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: Any) -> Any:
    """Handle item changes."""
    if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
      if value:
        self.setBrush(self._selected_brush)
        self.signals.node_clicked.emit(self)
      else:
        self.setBrush(self._default_brush)
        
    return super().itemChange(change, value)
    
  def mousePressEvent(self, event) -> None:
    """Handle mouse press events."""
    if event.button() == Qt.MouseButton.LeftButton:
      pos = event.pos()
      
      # Check if clicking on output port
      if self.output_port.contains(self.output_port.mapFromParent(pos)):
        self.signals.connection_started.emit(self)
        event.accept()
        return
        
      # Check if clicking on input port
      if self.input_port.contains(self.input_port.mapFromParent(pos)):
        self.signals.connection_finished.emit(self)
        event.accept()
        return
        
    super().mousePressEvent(event)
    
  def mouseDoubleClickEvent(self, event) -> None:
    """Handle double click events."""
    if event.button() == Qt.MouseButton.LeftButton:
      self.signals.node_clicked.emit(self)
      event.accept()
      return
      
    super().mouseDoubleClickEvent(event)
    
  def paint(self, painter: QPainter, option: Any, widget: Optional[QWidget] = None) -> None:
    """Custom paint method."""
    # Draw rounded rectangle
    painter.setPen(self.pen())
    painter.setBrush(self.brush())
    
    path = QPainterPath()
    path.addRoundedRect(self.rect(), 10, 10)
    painter.drawPath(path)
    
    # Draw port labels
    painter.setPen(QPen(Qt.GlobalColor.white, 1))
    painter.setFont(QFont("Arial", 8))
    
    # Input label
    painter.drawText(-5, 45, "In")
    
    # Output label
    painter.drawText(185, 45, "Out")
    
    # Draw info text at bottom
    info_text = self.get_info_text()
    if info_text:
      painter.drawText(10, 105, info_text)
      
  def get_info_text(self) -> str:
    """Get additional info text for the node."""
    return ""
    
  def get_input_port_pos(self) -> QPointF:
    """Get the position of the input port."""
    return self.pos() + QPointF(-5, 60)
    
  def get_output_port_pos(self) -> QPointF:
    """Get the position of the output port."""
    return self.pos() + QPointF(205, 60)
    
  def to_dict(self) -> Dict[str, Any]:
    """Convert node to dictionary for serialization."""
    return {
      "id": self.node_id,
      "type": self.__class__.__name__,
      "title": self.title,
      "position": {"x": self.pos().x(), "y": self.pos().y()},
      "properties": self.properties,
    }
    
  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "BaseNode":
    """Create node from dictionary."""
    # This should be implemented by subclasses
    raise NotImplementedError("Subclasses must implement from_dict")


class BasePropertyWidget(QWidget):
  """Base class for property widgets."""
  
  value_changed = pyqtSignal(str, object)  # property_name, value
  
  def __init__(self, node: BaseNode) -> None:
    super().__init__()
    self.node = node
    self._setup_ui()
    
  def _setup_ui(self) -> None:
    """Setup the property widget UI."""
    layout = QVBoxLayout(self)
    layout.setContentsMargins(0, 0, 0, 0)
    
    # Create property form
    self.form_layout = QFormLayout()
    layout.addLayout(self.form_layout)
    
    # Add properties
    self._add_properties()
    
  def _add_properties(self) -> None:
    """Add property fields. Override in subclasses."""
    pass
    
  def _add_text_property(self, name: str, default: str = "") -> QLineEdit:
    """Add a text property field."""
    line_edit = QLineEdit(str(default))
    line_edit.textChanged.connect(lambda text: self.value_changed.emit(name, text))
    self.form_layout.addRow(name.replace("_", " ").title() + ":", line_edit)
    return line_edit
    
  def _add_int_property(self, name: str, default: int = 0, min_val: int = 0, max_val: int = 999999) -> QSpinBox:
    """Add an integer property field."""
    spin_box = QSpinBox()
    spin_box.setRange(min_val, max_val)
    spin_box.setValue(default)
    spin_box.valueChanged.connect(lambda value: self.value_changed.emit(name, value))
    self.form_layout.addRow(name.replace("_", " ").title() + ":", spin_box)
    return spin_box
    
  def _add_float_property(self, name: str, default: float = 0.0, min_val: float = 0.0, max_val: float = 999999.0) -> QDoubleSpinBox:
    """Add a float property field."""
    spin_box = QDoubleSpinBox()
    spin_box.setRange(min_val, max_val)
    spin_box.setValue(default)
    spin_box.setDecimals(4)
    spin_box.valueChanged.connect(lambda value: self.value_changed.emit(name, value))
    self.form_layout.addRow(name.replace("_", " ").title() + ":", spin_box)
    return spin_box
    
  def _add_combo_property(self, name: str, options: List[str], default: str = "") -> QComboBox:
    """Add a combo box property field."""
    combo_box = QComboBox()
    combo_box.addItems(options)
    if default in options:
      combo_box.setCurrentText(default)
    combo_box.currentTextChanged.connect(lambda text: self.value_changed.emit(name, text))
    self.form_layout.addRow(name.replace("_", " ").title() + ":", combo_box)
    return combo_box