"""
Toolbar for the graph editor with layer nodes.
"""

from PyQt6.QtWidgets import (
  QToolBar,
  QVBoxLayout,
  QHBoxLayout,
  QWidget,
  QPushButton,
  QLabel,
  QScrollArea,
  QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QFont

from ...nodes.conv_nodes import Conv2DNode, MaxPool2DNode
from ...nodes.linear_nodes import LinearNode
from ...nodes.activation_nodes import ReLUNode, SigmoidNode
from ...nodes.utility_nodes import FlattenNode


class GraphToolbar(QWidget):
  """Toolbar containing draggable layer nodes."""
  
  node_requested = pyqtSignal(str, str)  # node_type, title
  
  def __init__(self) -> None:
    super().__init__()
    self._setup_ui()
    self._setup_buttons()
    
  def _setup_ui(self) -> None:
    """Setup the toolbar UI."""
    layout = QVBoxLayout(self)
    layout.setContentsMargins(5, 5, 5, 5)
    layout.setSpacing(5)
    
    # Title
    title_label = QLabel("Layers")
    title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
    layout.addWidget(title_label)
    
    # Scroll area for buttons
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    
    # Container widget for buttons
    container = QWidget()
    self.button_layout = QVBoxLayout(container)
    self.button_layout.setSpacing(3)
    self.button_layout.setContentsMargins(0, 0, 0, 0)
    
    scroll_area.setWidget(container)
    layout.addWidget(scroll_area)
    
    # Add stretch at bottom
    layout.addStretch()
    
  def _setup_buttons(self) -> None:
    """Setup layer category buttons."""
    # Convolutional Layers
    self._add_category("Convolutional")
    self._add_layer_button("Conv2D", "Conv2D", "2D Convolution")
    self._add_layer_button("MaxPool2D", "MaxPool2D", "2D Max Pooling")
    
    # Linear Layers
    self._add_category("Linear")
    self._add_layer_button("Linear", "Linear", "Fully Connected")
    
    # Activation Functions
    self._add_category("Activation")
    self._add_layer_button("ReLU", "ReLU", "Rectified Linear Unit")
    self._add_layer_button("Sigmoid", "Sigmoid", "Sigmoid Activation")
    
    # Utility Layers
    self._add_category("Utility")
    self._add_layer_button("Flatten", "Flatten", "Flatten Tensor")
    
  def _add_category(self, title: str) -> None:
    """Add a category separator."""
    separator = QFrame()
    separator.setFrameShape(QFrame.Shape.HLine)
    separator.setFrameShadow(QFrame.Shadow.Sunken)
    self.button_layout.addWidget(separator)
    
    category_label = QLabel(title)
    category_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
    category_label.setStyleSheet("color: #666; margin: 5px 0;")
    self.button_layout.addWidget(category_label)
    
  def _add_layer_button(self, node_type: str, title: str, tooltip: str) -> None:
    """Add a layer button to the toolbar."""
    button = QPushButton(title)
    button.setToolTip(tooltip)
    button.setStyleSheet("""
      QPushButton {
        background-color: #f0f0f0;
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 8px;
        text-align: left;
        font-size: 9px;
      }
      QPushButton:hover {
        background-color: #e0e0e0;
        border-color: #999;
      }
      QPushButton:pressed {
        background-color: #d0d0d0;
      }
    """)
    
    button.clicked.connect(lambda: self.node_requested.emit(node_type, title))
    self.button_layout.addWidget(button)
    
  def get_node_class(self, node_type: str) -> type:
    """Get the node class for the given type."""
    node_classes = {
      "Conv2D": Conv2DNode,
      "MaxPool2D": MaxPool2DNode,
      "Linear": LinearNode,
      "ReLU": ReLUNode,
      "Sigmoid": SigmoidNode,
      "Flatten": FlattenNode,
    }
    return node_classes.get(node_type, None)