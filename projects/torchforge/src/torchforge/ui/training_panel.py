"""
Training panel for model training configuration and monitoring.
"""

from typing import Dict, Any, Optional

from PyQt6.QtWidgets import (
  QWidget,
  QVBoxLayout,
  QHBoxLayout,
  QLabel,
  QPushButton,
  QComboBox,
  QSpinBox,
  QDoubleSpinBox,
  QProgressBar,
  QTextEdit,
  QGroupBox,
  QFormLayout,
  QCheckBox,
  QTabWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont

import pyqtgraph as pg


class TrainingPanel(QWidget):
  """Panel for training configuration and monitoring."""
  
  training_started = pyqtSignal(dict)
  training_stopped = pyqtSignal()
  training_paused = pyqtSignal()
  
  def __init__(self) -> None:
    super().__init__()
    self.is_training = False
    self.is_paused = False
    self._setup_ui()
    self._setup_plots()
    
  def _setup_ui(self) -> None:
    """Setup the training panel UI."""
    layout = QVBoxLayout(self)
    layout.setContentsMargins(5, 5, 5, 5)
    layout.setSpacing(5)
    
    # Create tab widget
    self.tab_widget = QTabWidget()
    layout.addWidget(self.tab_widget)
    
    # Configuration tab
    self.config_tab = self._create_config_tab()
    self.tab_widget.addTab(self.config_tab, "Configuration")
    
    # Monitoring tab
    self.monitor_tab = self._create_monitor_tab()
    self.tab_widget.addTab(self.monitor_tab, "Monitor")
    
    # Control buttons
    self._create_control_buttons(layout)
    
  def _create_config_tab(self) -> QWidget:
    """Create the configuration tab."""
    tab = QWidget()
    layout = QVBoxLayout(tab)
    
    # Dataset configuration
    dataset_group = QGroupBox("Dataset")
    dataset_layout = QFormLayout(dataset_group)
    
    self.dataset_combo = QComboBox()
    self.dataset_combo.addItems(["MNIST", "CIFAR-10", "Fashion-MNIST"])
    dataset_layout.addRow("Dataset:", self.dataset_combo)
    
    self.batch_size_spin = QSpinBox()
    self.batch_size_spin.setRange(1, 1024)
    self.batch_size_spin.setValue(32)
    dataset_layout.addRow("Batch Size:", self.batch_size_spin)
    
    layout.addWidget(dataset_group)
    
    # Training configuration
    training_group = QGroupBox("Training")
    training_layout = QFormLayout(training_group)
    
    self.epochs_spin = QSpinBox()
    self.epochs_spin.setRange(1, 1000)
    self.epochs_spin.setValue(10)
    training_layout.addRow("Epochs:", self.epochs_spin)
    
    self.learning_rate_spin = QDoubleSpinBox()
    self.learning_rate_spin.setRange(0.0001, 1.0)
    self.learning_rate_spin.setValue(0.001)
    self.learning_rate_spin.setDecimals(4)
    training_layout.addRow("Learning Rate:", self.learning_rate_spin)
    
    self.optimizer_combo = QComboBox()
    self.optimizer_combo.addItems(["Adam", "SGD", "AdamW", "RMSprop"])
    training_layout.addRow("Optimizer:", self.optimizer_combo)
    
    self.loss_combo = QComboBox()
    self.loss_combo.addItems(["CrossEntropyLoss", "MSELoss", "BCELoss"])
    training_layout.addRow("Loss Function:", self.loss_combo)
    
    layout.addWidget(training_group)
    
    # Advanced options
    advanced_group = QGroupBox("Advanced")
    advanced_layout = QFormLayout(advanced_group)
    
    self.device_combo = QComboBox()
    self.device_combo.addItems(["auto", "cpu", "cuda"])
    self.device_combo.setCurrentText("auto")
    advanced_layout.addRow("Device:", self.device_combo)
    
    self.early_stopping_check = QCheckBox("Enable Early Stopping")
    advanced_layout.addRow(self.early_stopping_check)
    
    self.save_checkpoints_check = QCheckBox("Save Checkpoints")
    self.save_checkpoints_check.setChecked(True)
    advanced_layout.addRow(self.save_checkpoints_check)
    
    layout.addWidget(advanced_group)
    
    layout.addStretch()
    return tab
    
  def _create_monitor_tab(self) -> QWidget:
    """Create the monitoring tab."""
    tab = QWidget()
    layout = QVBoxLayout(tab)
    
    # Progress info
    progress_group = QGroupBox("Training Progress")
    progress_layout = QVBoxLayout(progress_group)
    
    # Epoch progress
    epoch_layout = QHBoxLayout()
    epoch_layout.addWidget(QLabel("Epoch:"))
    self.epoch_label = QLabel("0/0")
    epoch_layout.addWidget(self.epoch_label)
    epoch_layout.addStretch()
    progress_layout.addLayout(epoch_layout)
    
    # Progress bar
    self.progress_bar = QProgressBar()
    progress_layout.addWidget(self.progress_bar)
    
    # Metrics
    metrics_layout = QHBoxLayout()
    
    loss_layout = QVBoxLayout()
    loss_layout.addWidget(QLabel("Loss:"))
    self.loss_label = QLabel("0.000")
    loss_layout.addWidget(self.loss_label)
    metrics_layout.addLayout(loss_layout)
    
    accuracy_layout = QVBoxLayout()
    accuracy_layout.addWidget(QLabel("Accuracy:"))
    self.accuracy_label = QLabel("0.00%")
    accuracy_layout.addWidget(self.accuracy_label)
    metrics_layout.addLayout(accuracy_layout)
    
    time_layout = QVBoxLayout()
    time_layout.addWidget(QLabel("Time:"))
    self.time_label = QLabel("00:00")
    time_layout.addWidget(self.time_label)
    metrics_layout.addLayout(time_layout)
    
    progress_layout.addLayout(metrics_layout)
    layout.addWidget(progress_group)
    
    # Plots
    plots_group = QGroupBox("Training Curves")
    plots_layout = QVBoxLayout(plots_group)
    
    # Create plot widget
    self.plot_widget = pg.PlotWidget(title="Loss and Accuracy")
    self.plot_widget.setLabel("left", "Value")
    self.plot_widget.setLabel("bottom", "Epoch")
    self.plot_widget.addLegend()
    self.plot_widget.showGrid(x=True, y=True)
    
    plots_layout.addWidget(self.plot_widget)
    layout.addWidget(plots_group)
    
    # Log output
    log_group = QGroupBox("Training Log")
    log_layout = QVBoxLayout(log_group)
    
    self.log_text = QTextEdit()
    self.log_text.setReadOnly(True)
    self.log_text.setMaximumHeight(150)
    log_layout.addWidget(self.log_text)
    
    layout.addWidget(log_group)
    
    return tab
    
  def _create_control_buttons(self, layout: QVBoxLayout) -> None:
    """Create training control buttons."""
    button_layout = QHBoxLayout()
    
    self.start_button = QPushButton("Start Training")
    self.start_button.clicked.connect(self._on_start_training)
    button_layout.addWidget(self.start_button)
    
    self.pause_button = QPushButton("Pause")
    self.pause_button.clicked.connect(self._on_pause_training)
    self.pause_button.setEnabled(False)
    button_layout.addWidget(self.pause_button)
    
    self.stop_button = QPushButton("Stop")
    self.stop_button.clicked.connect(self._on_stop_training)
    self.stop_button.setEnabled(False)
    button_layout.addWidget(self.stop_button)
    
    layout.addLayout(button_layout)
    
  def _setup_plots(self) -> None:
    """Setup the training plots."""
    # Create plot curves
    self.loss_curve = self.plot_widget.plot(
      pen=pg.mkPen("r", width=2), name="Loss"
    )
    self.accuracy_curve = self.plot_widget.plot(
      pen=pg.mkPen("b", width=2), name="Accuracy"
    )
    
    # Initialize data
    self.epoch_data = []
    self.loss_data = []
    self.accuracy_data = []
    
  def _on_start_training(self) -> None:
    """Handle start training button click."""
    config = self.get_training_config()
    self.training_started.emit(config)
    
    self.is_training = True
    self.is_paused = False
    
    # Update button states
    self.start_button.setEnabled(False)
    self.pause_button.setEnabled(True)
    self.stop_button.setEnabled(True)
    
    # Switch to monitor tab
    self.tab_widget.setCurrentIndex(1)
    
    # Clear previous data
    self._clear_training_data()
    
    # Log start
    self.log_message("Training started...")
    
  def _on_pause_training(self) -> None:
    """Handle pause training button click."""
    if self.is_paused:
      self.training_paused.emit()
      self.pause_button.setText("Pause")
      self.is_paused = False
      self.log_message("Training resumed...")
    else:
      self.pause_button.setText("Resume")
      self.is_paused = True
      self.log_message("Training paused...")
      
  def _on_stop_training(self) -> None:
    """Handle stop training button click."""
    self.training_stopped.emit()
    
    self.is_training = False
    self.is_paused = False
    
    # Update button states
    self.start_button.setEnabled(True)
    self.pause_button.setEnabled(False)
    self.pause_button.setText("Pause")
    self.stop_button.setEnabled(False)
    
    self.log_message("Training stopped.")
    
  def get_training_config(self) -> Dict[str, Any]:
    """Get the current training configuration."""
    return {
      "dataset": self.dataset_combo.currentText(),
      "batch_size": self.batch_size_spin.value(),
      "epochs": self.epochs_spin.value(),
      "learning_rate": self.learning_rate_spin.value(),
      "optimizer": self.optimizer_combo.currentText(),
      "loss_function": self.loss_combo.currentText(),
      "device": self.device_combo.currentText(),
      "early_stopping": self.early_stopping_check.isChecked(),
      "save_checkpoints": self.save_checkpoints_check.isChecked(),
    }
    
  def update_progress(self, epoch: int, total_epochs: int, loss: float, accuracy: float) -> None:
    """Update training progress."""
    # Update labels
    self.epoch_label.setText(f"{epoch}/{total_epochs}")
    self.loss_label.setText(f"{loss:.4f}")
    self.accuracy_label.setText(f"{accuracy:.2f}%")
    
    # Update progress bar
    progress = int((epoch / total_epochs) * 100)
    self.progress_bar.setValue(progress)
    
    # Update plot data
    self.epoch_data.append(epoch)
    self.loss_data.append(loss)
    self.accuracy_data.append(accuracy)
    
    self.loss_curve.setData(self.epoch_data, self.loss_data)
    self.accuracy_curve.setData(self.epoch_data, self.accuracy_data)
    
  def log_message(self, message: str) -> None:
    """Add a message to the training log."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    self.log_text.append(f"[{timestamp}] {message}")
        
  def _clear_training_data(self) -> None:
    """Clear previous training data."""
    self.epoch_data.clear()
    self.loss_data.clear()
    self.accuracy_data.clear()
    
    self.loss_curve.setData([], [])
    self.accuracy_curve.setData([], [])
    
    self.epoch_label.setText("0/0")
    self.loss_label.setText("0.000")
    self.accuracy_label.setText("0.00%")
    self.progress_bar.setValue(0)
    
    self.log_text.clear()