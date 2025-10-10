"""
Experiment panel for managing and comparing experiments.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from PyQt6.QtWidgets import (
  QWidget,
  QVBoxLayout,
  QHBoxLayout,
  QLabel,
  QPushButton,
  QListWidget,
  QListWidgetItem,
  QTableWidget,
  QTableWidgetItem,
  QHeaderView,
  QSplitter,
  QGroupBox,
  QTextEdit,
  QComboBox,
  QLineEdit,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

import pyqtgraph as pg


class ExperimentPanel(QWidget):
  """Panel for managing and comparing experiments."""
  
  experiment_selected = pyqtSignal(str)  # experiment_id
  compare_requested = pyqtSignal(list)  # experiment_ids
  
  def __init__(self) -> None:
    super().__init__()
    self.experiments: Dict[str, Dict[str, Any]] = {}
    self._setup_ui()
    
  def _setup_ui(self) -> None:
    """Setup the experiment panel UI."""
    layout = QVBoxLayout(self)
    layout.setContentsMargins(5, 5, 5, 5)
    layout.setSpacing(5)
    
    # Title
    title_label = QLabel("Experiments")
    title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
    layout.addWidget(title_label)
    
    # Create splitter
    splitter = QSplitter(Qt.Orientation.Horizontal)
    layout.addWidget(splitter)
    
    # Left side - Experiment list
    left_widget = self._create_experiment_list()
    splitter.addWidget(left_widget)
    
    # Right side - Experiment details
    right_widget = self._create_experiment_details()
    splitter.addWidget(right_widget)
    
    # Set splitter proportions
    splitter.setSizes([300, 500])
    
  def _create_experiment_list(self) -> QWidget:
    """Create the experiment list widget."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    
    # Controls
    controls_layout = QHBoxLayout()
    
    self.new_experiment_button = QPushButton("New")
    self.new_experiment_button.clicked.connect(self._on_new_experiment)
    controls_layout.addWidget(self.new_experiment_button)
    
    self.delete_experiment_button = QPushButton("Delete")
    self.delete_experiment_button.clicked.connect(self._on_delete_experiment)
    self.delete_experiment_button.setEnabled(False)
    controls_layout.addWidget(self.delete_experiment_button)
    
    self.compare_button = QPushButton("Compare")
    self.compare_button.clicked.connect(self._on_compare_experiments)
    self.compare_button.setEnabled(False)
    controls_layout.addWidget(self.compare_button)
    
    controls_layout.addStretch()
    layout.addLayout(controls_layout)
    
    # Search
    self.search_edit = QLineEdit()
    self.search_edit.setPlaceholderText("Search experiments...")
    self.search_edit.textChanged.connect(self._filter_experiments)
    layout.addWidget(self.search_edit)
    
    # Experiment list
    self.experiment_list = QListWidget()
    self.experiment_list.itemSelectionChanged.connect(self._on_experiment_selected)
    self.experiment_list.itemChanged.connect(self._on_experiment_renamed)
    layout.addWidget(self.experiment_list)
    
    return widget
    
  def _create_experiment_details(self) -> QWidget:
    """Create the experiment details widget."""
    widget = QWidget()
    layout = QVBoxLayout(widget)
    
    # Experiment info
    info_group = QGroupBox("Experiment Information")
    info_layout = QFormLayout(info_group)
    
    self.experiment_id_label = QLabel("-")
    info_layout.addRow("ID:", self.experiment_id_label)
    
    self.experiment_date_label = QLabel("-")
    info_layout.addRow("Date:", self.experiment_date_label)
    
    self.experiment_status_label = QLabel("-")
    info_layout.addRow("Status:", self.experiment_status_label)
    
    layout.addWidget(info_group)
    
    # Hyperparameters
    params_group = QGroupBox("Hyperparameters")
    params_layout = QVBoxLayout(params_group)
    
    self.params_table = QTableWidget()
    self.params_table.setColumnCount(2)
    self.params_table.setHorizontalHeaderLabels(["Parameter", "Value"])
    self.params_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    self.params_table.verticalHeader().setVisible(False)
    params_layout.addWidget(self.params_table)
    
    layout.addWidget(params_group)
    
    # Results
    results_group = QGroupBox("Results")
    results_layout = QVBoxLayout(results_group)
    
    # Results table
    self.results_table = QTableWidget()
    self.results_table.setColumnCount(2)
    self.results_table.setHorizontalHeaderLabels(["Metric", "Value"])
    self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    self.results_table.verticalHeader().setVisible(False)
    results_layout.addWidget(self.results_table)
    
    # Comparison plot
    self.comparison_plot = pg.PlotWidget(title="Performance Comparison")
    self.comparison_plot.setLabel("left", "Value")
    self.comparison_plot.setLabel("bottom", "Epoch")
    self.comparison_plot.addLegend()
    self.comparison_plot.showGrid(x=True, y=True)
    results_layout.addWidget(self.comparison_plot)
    
    layout.addWidget(results_group)
    
    # Notes
    notes_group = QGroupBox("Notes")
    notes_layout = QVBoxLayout(notes_group)
    
    self.notes_edit = QTextEdit()
    self.notes_edit.setMaximumHeight(100)
    self.notes_edit.textChanged.connect(self._on_notes_changed)
    notes_layout.addWidget(self.notes_edit)
    
    layout.addWidget(notes_group)
    
    return widget
    
  def add_experiment(self, experiment_id: str, config: Dict[str, Any]) -> None:
    """Add a new experiment."""
    experiment = {
      "id": experiment_id,
      "name": f"Experiment {len(self.experiments) + 1}",
      "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      "status": "Created",
      "config": config,
      "results": {},
      "notes": "",
      "history": {"loss": [], "accuracy": []},
    }
    
    self.experiments[experiment_id] = experiment
    
    # Add to list
    item = QListWidgetItem(experiment["name"])
    item.setData(Qt.ItemDataRole.UserRole, experiment_id)
    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
    self.experiment_list.addItem(item)
    
  def update_experiment_status(self, experiment_id: str, status: str) -> None:
    """Update experiment status."""
    if experiment_id in self.experiments:
      self.experiments[experiment_id]["status"] = status
      if self._get_selected_experiment_id() == experiment_id:
        self.experiment_status_label.setText(status)
        
  def update_experiment_results(self, experiment_id: str, results: Dict[str, Any]) -> None:
    """Update experiment results."""
    if experiment_id in self.experiments:
      self.experiments[experiment_id]["results"].update(results)
      if self._get_selected_experiment_id() == experiment_id:
        self._update_results_display(results)
        
  def add_training_history(self, experiment_id: str, epoch: int, loss: float, accuracy: float) -> None:
    """Add training history data."""
    if experiment_id in self.experiments:
      history = self.experiments[experiment_id]["history"]
      history["loss"].append((epoch, loss))
      history["accuracy"].append((epoch, accuracy))
      
      if self._get_selected_experiment_id() == experiment_id:
        self._update_comparison_plot()
        
  def _on_new_experiment(self) -> None:
    """Handle new experiment button click."""
    import uuid
    experiment_id = str(uuid.uuid4())[:8]
    
    # Default config
    config = {
      "dataset": "MNIST",
      "batch_size": 32,
      "epochs": 10,
      "learning_rate": 0.001,
      "optimizer": "Adam",
    }
    
    self.add_experiment(experiment_id, config)
    
  def _on_delete_experiment(self) -> None:
    """Handle delete experiment button click."""
    current_item = self.experiment_list.currentItem()
    if current_item:
      experiment_id = current_item.data(Qt.ItemDataRole.UserRole)
      if experiment_id in self.experiments:
        del self.experiments[experiment_id]
        
      row = self.experiment_list.row(current_item)
      self.experiment_list.takeItem(row)
      
  def _on_compare_experiments(self) -> None:
    """Handle compare experiments button click."""
    selected_items = self.experiment_list.selectedItems()
    experiment_ids = [
      item.data(Qt.ItemDataRole.UserRole) for item in selected_items
    ]
    
    if len(experiment_ids) >= 2:
      self.compare_requested.emit(experiment_ids)
      
  def _on_experiment_selected(self) -> None:
    """Handle experiment selection."""
    current_item = self.experiment_list.currentItem()
    if current_item:
      experiment_id = current_item.data(Qt.ItemDataRole.UserRole)
      self.experiment_selected.emit(experiment_id)
      self._display_experiment_details(experiment_id)
      
      # Update button states
      self.delete_experiment_button.setEnabled(True)
      selected_count = len(self.experiment_list.selectedItems())
      self.compare_button.setEnabled(selected_count >= 2)
    else:
      self._clear_experiment_details()
      self.delete_experiment_button.setEnabled(False)
      self.compare_button.setEnabled(False)
      
  def _on_experiment_renamed(self, item: QListWidgetItem) -> None:
    """Handle experiment rename."""
    experiment_id = item.data(Qt.ItemDataRole.UserRole)
    if experiment_id in self.experiments:
      self.experiments[experiment_id]["name"] = item.text()
      
  def _on_notes_changed(self) -> None:
    """Handle notes text change."""
    experiment_id = self._get_selected_experiment_id()
    if experiment_id in self.experiments:
      self.experiments[experiment_id]["notes"] = self.notes_edit.toPlainText()
      
  def _filter_experiments(self, text: str) -> None:
    """Filter experiments based on search text."""
    for i in range(self.experiment_list.count()):
      item = self.experiment_list.item(i)
      item.setHidden(text.lower() not in item.text().lower())
      
  def _get_selected_experiment_id(self) -> Optional[str]:
    """Get the currently selected experiment ID."""
    current_item = self.experiment_list.currentItem()
    if current_item:
      return current_item.data(Qt.ItemDataRole.UserRole)
    return None
    
  def _display_experiment_details(self, experiment_id: str) -> None:
    """Display details for the selected experiment."""
    if experiment_id not in self.experiments:
      return
      
    experiment = self.experiments[experiment_id]
    
    # Update info
    self.experiment_id_label.setText(experiment_id)
    self.experiment_date_label.setText(experiment["date"])
    self.experiment_status_label.setText(experiment["status"])
    
    # Update hyperparameters
    self._update_params_display(experiment["config"])
    
    # Update results
    self._update_results_display(experiment["results"])
    
    # Update notes
    self.notes_edit.setPlainText(experiment["notes"])
    
    # Update comparison plot
    self._update_comparison_plot()
    
  def _update_params_display(self, config: Dict[str, Any]) -> None:
    """Update the hyperparameters table."""
    self.params_table.setRowCount(0)
    
    for key, value in config.items():
      row = self.params_table.rowCount()
      self.params_table.insertRow(row)
      
      key_item = QTableWidgetItem(str(key).replace("_", " ").title())
      value_item = QTableWidgetItem(str(value))
      
      self.params_table.setItem(row, 0, key_item)
      self.params_table.setItem(row, 1, value_item)
      
  def _update_results_display(self, results: Dict[str, Any]) -> None:
    """Update the results table."""
    self.results_table.setRowCount(0)
    
    for key, value in results.items():
      row = self.results_table.rowCount()
      self.results_table.insertRow(row)
      
      key_item = QTableWidgetItem(str(key).replace("_", " ").title())
      value_item = QTableWidgetItem(str(value))
      
      self.results_table.setItem(row, 0, key_item)
      self.results_table.setItem(row, 1, value_item)
      
  def _update_comparison_plot(self) -> None:
    """Update the comparison plot."""
    self.comparison_plot.clear()
    
    experiment_id = self._get_selected_experiment_id()
    if experiment_id not in self.experiments:
      return
      
    history = self.experiments[experiment_id]["history"]
    
    # Plot loss
    if history["loss"]:
      epochs, losses = zip(*history["loss"])
      self.comparison_plot.plot(
        epochs, losses, pen=pg.mkPen("r", width=2), name="Loss"
      )
      
    # Plot accuracy
    if history["accuracy"]:
      epochs, accuracies = zip(*history["accuracy"])
      self.comparison_plot.plot(
        epochs, accuracies, pen=pg.mkPen("b", width=2), name="Accuracy"
      )
      
  def _clear_experiment_details(self) -> None:
    """Clear the experiment details display."""
    self.experiment_id_label.setText("-")
    self.experiment_date_label.setText("-")
    self.experiment_status_label.setText("-")
    
    self.params_table.setRowCount(0)
    self.results_table.setRowCount(0)
    
    self.notes_edit.clear()
    self.comparison_plot.clear()