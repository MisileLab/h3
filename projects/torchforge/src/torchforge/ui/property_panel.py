"""
Property panel for editing node properties.
"""

from typing import Any, Dict, Optional

from PyQt6.QtWidgets import (
  QWidget,
  QVBoxLayout,
  QHBoxLayout,
  QLabel,
  QLineEdit,
  QSpinBox,
  QDoubleSpinBox,
  QComboBox,
  QPushButton,
  QScrollArea,
  QFrame,
  QGroupBox,
  QFormLayout,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from ..graph_editor.node import BaseNode


class PropertyPanel(QWidget):
  """Panel for displaying and editing node properties."""
  
  property_changed = pyqtSignal(str, object, object)  # node_id, property_name, value
  
  def __init__(self) -> None:
    super().__init__()
    self.current_node: Optional[BaseNode] = None
    self.property_widgets: Dict[str, QWidget] = {}
    self._setup_ui()
    
  def _setup_ui(self) -> None:
    """Setup the property panel UI."""
    layout = QVBoxLayout(self)
    layout.setContentsMargins(5, 5, 5, 5)
    layout.setSpacing(5)
    
    # Title
    title_label = QLabel("Properties")
    title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
    layout.addWidget(title_label)
    
    # Scroll area for properties
    self.scroll_area = QScrollArea()
    self.scroll_area.setWidgetResizable(True)
    self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    
    # Container for properties
    self.container = QWidget()
    self.container_layout = QVBoxLayout(self.container)
    self.container_layout.setSpacing(10)
    self.container_layout.setContentsMargins(0, 0, 0, 0)
    
    self.scroll_area.setWidget(self.container)
    layout.addWidget(self.scroll_area)
    
    # Default message
    self._show_default_message()
    
  def _show_default_message(self) -> None:
    """Show default message when no node is selected."""
    self._clear_properties()
    
    message_label = QLabel("Select a node to view its properties")
    message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    message_label.setStyleSheet("color: #999; font-style: italic; padding: 20px;")
    self.container_layout.addWidget(message_label)
    
  def set_node(self, node: Optional[BaseNode]) -> None:
    """Set the current node to display properties for."""
    # Disconnect previous node signals
    if self.current_node:
      self.current_node.signals.property_changed.disconnect(self._on_property_changed)
      
    self.current_node = node
    
    if node is None:
      self._show_default_message()
      return
      
    # Connect new node signals
    node.signals.property_changed.connect(self._on_property_changed)
    
    # Display node properties
    self._display_node_properties(node)
    
  def _display_node_properties(self, node: BaseNode) -> None:
    """Display properties for the given node."""
    self._clear_properties()
    
    # Node info
    info_group = QGroupBox("Node Information")
    info_layout = QFormLayout(info_group)
    
    # Node ID
    id_label = QLabel(node.node_id)
    id_label.setWordWrap(True)
    info_layout.addRow("ID:", id_label)
    
    # Node type
    type_label = QLabel(node.__class__.__name__)
    info_layout.addRow("Type:", type_label)
    
    # Layer type
    layer_type_label = QLabel(node.get_layer_type())
    info_layout.addRow("Layer:", layer_type_label)
    
    self.container_layout.addWidget(info_group)
    
    # Node-specific properties
    properties_widget = node.create_property_widget()
    if properties_widget:
      self.container_layout.addWidget(properties_widget)
      
    # Add stretch at bottom
    self.container_layout.addStretch()
    
  def _clear_properties(self) -> None:
    """Clear all property widgets."""
    # Remove all widgets from container
    while self.container_layout.count():
      child = self.container_layout.takeAt(0)
      if child.widget():
        child.widget().deleteLater()
        
    self.property_widgets.clear()
    
  def _on_property_changed(self, property_name: str, value: Any) -> None:
    """Handle property change from node."""
    if self.current_node:
      self.property_changed.emit(self.current_node.node_id, property_name, value)


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
    
  def _add_combo_property(self, name: str, options: list[str], default: str = "") -> QComboBox:
    """Add a combo box property field."""
    combo_box = QComboBox()
    combo_box.addItems(options)
    if default in options:
      combo_box.setCurrentText(default)
    combo_box.currentTextChanged.connect(lambda text: self.value_changed.emit(name, text))
    self.form_layout.addRow(name.replace("_", " ").title() + ":", combo_box)
    return combo_box