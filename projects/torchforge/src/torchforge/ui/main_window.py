"""
Main application window for TorchForge.
"""

from PyQt6.QtWidgets import (
  QMainWindow,
  QWidget,
  QHBoxLayout,
  QVBoxLayout,
  QSplitter,
  QMenuBar,
  QStatusBar,
  QDockWidget,
  QToolBar,
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QIcon

from .graph_editor.canvas import GraphCanvas
from .property_panel import PropertyPanel
from .training_panel import TrainingPanel
from .experiment_panel import ExperimentPanel


class MainWindow(QMainWindow):
  """Main application window."""
  
  def __init__(self) -> None:
    super().__init__()
    self.setWindowTitle("TorchForge - Visual PyTorch Model Builder")
    self.setMinimumSize(1200, 800)
    self.resize(1600, 1000)
    
    self._setup_ui()
    self._setup_menu()
    self._setup_toolbar()
    self._setup_statusbar()
    
  def _setup_ui(self) -> None:
    """Setup the main UI layout."""
    # Central widget with splitter
    central_widget = QWidget()
    self.setCentralWidget(central_widget)
    
    # Main horizontal splitter
    main_splitter = QSplitter(Qt.Orientation.Horizontal)
    central_layout = QHBoxLayout(central_widget)
    central_layout.setContentsMargins(0, 0, 0, 0)
    central_layout.addWidget(main_splitter)
    
    # Left panel - Toolbox
    self.toolbox_dock = QDockWidget("Toolbox", self)
    self.toolbox_dock.setAllowedAreas(
      Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
    )
    self.toolbox_dock.setFixedWidth(200)
    main_splitter.addWidget(self.toolbox_dock)
    
    # Center - Graph Canvas
    self.graph_canvas = GraphCanvas()
    main_splitter.addWidget(self.graph_canvas)
    
    # Right panel - Properties
    self.property_panel = PropertyPanel()
    property_dock = QDockWidget("Properties", self)
    property_dock.setAllowedAreas(
      Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
    )
    property_dock.setWidget(self.property_panel)
    property_dock.setFixedWidth(300)
    main_splitter.addWidget(property_dock)
    
    # Set splitter proportions
    main_splitter.setSizes([200, 1000, 300])
    
    # Bottom - Training Panel
    self.training_panel = TrainingPanel()
    training_dock = QDockWidget("Training", self)
    training_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
    training_dock.setWidget(self.training_panel)
    self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, training_dock)
    
    # Experiment Panel (optional)
    self.experiment_panel = ExperimentPanel()
    experiment_dock = QDockWidget("Experiments", self)
    experiment_dock.setAllowedAreas(
      Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
    )
    experiment_dock.setWidget(self.experiment_panel)
    self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, experiment_dock)
    
    # Initially hide experiment panel
    experiment_dock.hide()
    
  def _setup_menu(self) -> None:
    """Setup the menu bar."""
    menubar = self.menuBar()
    
    # File menu
    file_menu = menubar.addMenu("&File")
    
    new_action = QAction("&New Project", self)
    new_action.setShortcut("Ctrl+N")
    new_action.setStatusTip("Create a new project")
    file_menu.addAction(new_action)
    
    open_action = QAction("&Open Project", self)
    open_action.setShortcut("Ctrl+O")
    open_action.setStatusTip("Open an existing project")
    file_menu.addAction(open_action)
    
    save_action = QAction("&Save Project", self)
    save_action.setShortcut("Ctrl+S")
    save_action.setStatusTip("Save the current project")
    file_menu.addAction(save_action)
    
    file_menu.addSeparator()
    
    export_code_action = QAction("Export &Code", self)
    export_code_action.setShortcut("Ctrl+E")
    export_code_action.setStatusTip("Export model as PyTorch code")
    file_menu.addAction(export_code_action)
    
    export_model_action = QAction("Export &Model", self)
    export_model_action.setStatusTip("Export trained model")
    file_menu.addAction(export_model_action)
    
    file_menu.addSeparator()
    
    exit_action = QAction("E&xit", self)
    exit_action.setShortcut("Ctrl+Q")
    exit_action.setStatusTip("Exit the application")
    exit_action.triggered.connect(self.close)
    file_menu.addAction(exit_action)
    
    # Edit menu
    edit_menu = menubar.addMenu("&Edit")
    
    undo_action = QAction("&Undo", self)
    undo_action.setShortcut("Ctrl+Z")
    edit_menu.addAction(undo_action)
    
    redo_action = QAction("&Redo", self)
    redo_action.setShortcut("Ctrl+Y")
    edit_menu.addAction(redo_action)
    
    edit_menu.addSeparator()
    
    delete_action = QAction("&Delete", self)
    delete_action.setShortcut("Del")
    edit_menu.addAction(delete_action)
    
    # View menu
    view_menu = menubar.addMenu("&View")
    
    view_menu.addAction(self.toolbox_dock.toggleViewAction())
    view_menu.addAction(self.property_panel.parent().toggleViewAction())
    view_menu.addAction(self.training_panel.parent().toggleViewAction())
    view_menu.addAction(self.experiment_panel.parent().toggleViewAction())
    
    # Model menu
    model_menu = menubar.addMenu("&Model")
    
    validate_action = QAction("&Validate Graph", self)
    validate_action.setShortcut("F5")
    validate_action.setStatusTip("Validate the model graph")
    model_menu.addAction(validate_action)
    
    generate_code_action = QAction("&Generate Code", self)
    generate_code_action.setShortcut("F6")
    generate_code_action.setStatusTip("Generate PyTorch code from graph")
    model_menu.addAction(generate_code_action)
    
    # Train menu
    train_menu = menubar.addMenu("&Train")
    
    start_training_action = QAction("&Start Training", self)
    start_training_action.setShortcut("F9")
    start_training_action.setStatusTip("Start model training")
    train_menu.addAction(start_training_action)
    
    stop_training_action = QAction("St&op Training", self)
    stop_training_action.setShortcut("F10")
    stop_training_action.setStatusTip("Stop model training")
    train_menu.addAction(stop_training_action)
    
    # Help menu
    help_menu = menubar.addMenu("&Help")
    
    about_action = QAction("&About", self)
    about_action.setStatusTip("Show information about TorchForge")
    help_menu.addAction(about_action)
    
  def _setup_toolbar(self) -> None:
    """Setup the main toolbar."""
    toolbar = QToolBar("Main Toolbar")
    toolbar.setMovable(False)
    self.addToolBar(toolbar)
    
    # Add actions to toolbar
    new_action = QAction("New", self)
    toolbar.addAction(new_action)
    
    open_action = QAction("Open", self)
    toolbar.addAction(open_action)
    
    save_action = QAction("Save", self)
    toolbar.addAction(save_action)
    
    toolbar.addSeparator()
    
    undo_action = QAction("Undo", self)
    toolbar.addAction(undo_action)
    
    redo_action = QAction("Redo", self)
    toolbar.addAction(redo_action)
    
    toolbar.addSeparator()
    
    validate_action = QAction("Validate", self)
    toolbar.addAction(validate_action)
    
    generate_action = QAction("Generate", self)
    toolbar.addAction(generate_action)
    
    toolbar.addSeparator()
    
    train_action = QAction("Train", self)
    toolbar.addAction(train_action)
    
  def _setup_statusbar(self) -> None:
    """Setup the status bar."""
    self.status_bar = QStatusBar()
    self.setStatusBar(self.status_bar)
    self.status_bar.showMessage("Ready")