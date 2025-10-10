"""
Training manager for PyTorch models.
"""

import time
from typing import Dict, Any, Optional, Callable
import threading

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from .model_builder import ModelBuilder


class Trainer:
  """Manages model training with progress tracking."""
  
  def __init__(self) -> None:
    self.model: Optional[nn.Module] = None
    self.device: torch.device = torch.device("cpu")
    self.optimizer: Optional[optim.Optimizer] = None
    self.criterion: Optional[nn.Module] = None
    self.train_loader: Optional[DataLoader] = None
    self.val_loader: Optional[DataLoader] = None
    
    # Training state
    self.is_training = False
    self.is_paused = False
    self.should_stop = False
    self.current_epoch = 0
    self.total_epochs = 0
    
    # Callbacks
    self.progress_callback: Optional[Callable] = None
    self.log_callback: Optional[Callable] = None
    
    # Training history
    self.history: Dict[str, list] = {
      "train_loss": [],
      "train_accuracy": [],
      "val_loss": [],
      "val_accuracy": [],
    }
    
  def setup_model(self, model_builder: ModelBuilder, config: Dict[str, Any]) -> str:
    """Setup the model from model builder."""
    try:
      # Generate PyTorch code
      code = model_builder.generate_pytorch_code()
      
      # Execute code to create model
      namespace = {}
      exec(code, namespace)
      
      # Get the model class (assume it's the last class defined)
      model_classes = [obj for name, obj in namespace.items() 
                      if isinstance(obj, type) and issubclass(obj, nn.Module) and obj != nn.Module]
      
      if not model_classes:
        raise ValueError("No model class found in generated code")
        
      model_class = model_classes[0]
      self.model = model_class()
      
      # Setup device
      device_name = config.get("device", "auto")
      if device_name == "auto":
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      else:
        self.device = torch.device(device_name)
        
      self.model.to(self.device)
      
      return f"Model loaded on {self.device}"
      
    except Exception as e:
      raise ValueError(f"Failed to setup model: {e}")
      
  def setup_training(self, config: Dict[str, Any]) -> None:
    """Setup training components."""
    if self.model is None:
      raise ValueError("Model not setup")
      
    # Setup optimizer
    optimizer_name = config.get("optimizer", "Adam")
    learning_rate = config.get("learning_rate", 0.001)
    
    if optimizer_name == "Adam":
      self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
      self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
    elif optimizer_name == "AdamW":
      self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSprop":
      self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
    else:
      raise ValueError(f"Unknown optimizer: {optimizer_name}")
      
    # Setup loss function
    loss_name = config.get("loss_function", "CrossEntropyLoss")
    if loss_name == "CrossEntropyLoss":
      self.criterion = nn.CrossEntropyLoss()
    elif loss_name == "MSELoss":
      self.criterion = nn.MSELoss()
    elif loss_name == "BCELoss":
      self.criterion = nn.BCELoss()
    else:
      raise ValueError(f"Unknown loss function: {loss_name}")
      
    # Setup data loaders
    self._setup_data_loaders(config)
    
  def _setup_data_loaders(self, config: Dict[str, Any]) -> None:
    """Setup training and validation data loaders."""
    dataset_name = config.get("dataset", "MNIST")
    batch_size = config.get("batch_size", 32)
    
    # Define transforms
    if dataset_name in ["MNIST", "Fashion-MNIST"]:
      transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
      ])
    elif dataset_name == "CIFAR-10":
      transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])
    else:
      transform = transforms.ToTensor()
      
    # Load datasets
    if dataset_name == "MNIST":
      train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
      )
      val_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
      )
    elif dataset_name == "Fashion-MNIST":
      train_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
      )
      val_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
      )
    elif dataset_name == "CIFAR-10":
      train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
      )
      val_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
      )
    else:
      raise ValueError(f"Unknown dataset: {dataset_name}")
      
    self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
  def start_training(self, config: Dict[str, Any]) -> None:
    """Start training in a separate thread."""
    if self.is_training:
      raise ValueError("Training already in progress")
      
    self.total_epochs = config.get("epochs", 10)
    self.current_epoch = 0
    self.should_stop = False
    self.is_paused = False
    
    # Start training thread
    training_thread = threading.Thread(target=self._training_loop)
    training_thread.daemon = True
    training_thread.start()
    
  def _training_loop(self) -> None:
    """Main training loop."""
    self.is_training = True
    
    try:
      for epoch in range(self.current_epoch, self.total_epochs):
        if self.should_stop:
          break
          
        # Wait if paused
        while self.is_paused and not self.should_stop:
          time.sleep(0.1)
          
        if self.should_stop:
          break
          
        self.current_epoch = epoch + 1
        
        # Training phase
        train_loss, train_accuracy = self._train_epoch()
        
        # Validation phase
        val_loss, val_accuracy = self._validate_epoch()
        
        # Update history
        self.history["train_loss"].append(train_loss)
        self.history["train_accuracy"].append(train_accuracy)
        self.history["val_loss"].append(val_loss)
        self.history["val_accuracy"].append(val_accuracy)
        
        # Call progress callback
        if self.progress_callback:
          self.progress_callback(epoch + 1, self.total_epochs, train_loss, train_accuracy)
          
        # Log progress
        if self.log_callback:
          self.log_callback(f"Epoch {epoch + 1}/{self.total_epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
                          
    except Exception as e:
      if self.log_callback:
        self.log_callback(f"Training error: {e}")
    finally:
      self.is_training = False
      
  def _train_epoch(self) -> tuple[float, float]:
    """Train for one epoch."""
    if self.model is None or self.train_loader is None:
      raise ValueError("Model or train loader not setup")
      
    self.model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(self.train_loader):
      if self.should_stop:
        break
        
      data, target = data.to(self.device), target.to(self.device)
      
      self.optimizer.zero_grad()
      output = self.model(data)
      loss = self.criterion(output, target)
      loss.backward()
      self.optimizer.step()
      
      total_loss += loss.item()
      
      # Calculate accuracy
      if hasattr(self.criterion, '__class__') and 'CrossEntropy' in self.criterion.__class__.__name__:
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
    avg_loss = total_loss / len(self.train_loader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy
    
  def _validate_epoch(self) -> tuple[float, float]:
    """Validate for one epoch."""
    if self.model is None or self.val_loader is None:
      raise ValueError("Model or validation loader not setup")
      
    self.model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
      for data, target in self.val_loader:
        if self.should_stop:
          break
          
        data, target = data.to(self.device), target.to(self.device)
        output = self.model(data)
        loss = self.criterion(output, target)
        
        total_loss += loss.item()
        
        # Calculate accuracy
        if hasattr(self.criterion, '__class__') and 'CrossEntropy' in self.criterion.__class__.__name__:
          pred = output.argmax(dim=1, keepdim=True)
          correct += pred.eq(target.view_as(pred)).sum().item()
          total += target.size(0)
          
    avg_loss = total_loss / len(self.val_loader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy
    
  def pause_training(self) -> None:
    """Pause training."""
    self.is_paused = True
    
  def resume_training(self) -> None:
    """Resume training."""
    self.is_paused = False
    
  def stop_training(self) -> None:
    """Stop training."""
    self.should_stop = True
    
  def save_model(self, filepath: str) -> None:
    """Save the trained model."""
    if self.model is None:
      raise ValueError("No model to save")
      
    torch.save({
      "model_state_dict": self.model.state_dict(),
      "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
      "history": self.history,
      "current_epoch": self.current_epoch,
    }, filepath)
    
  def load_model(self, filepath: str) -> None:
    """Load a trained model."""
    checkpoint = torch.load(filepath, map_location=self.device)
    
    if self.model is None:
      raise ValueError("Model not setup")
      
    self.model.load_state_dict(checkpoint["model_state_dict"])
    
    if self.optimizer and checkpoint["optimizer_state_dict"]:
      self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
      
    self.history = checkpoint.get("history", self.history)
    self.current_epoch = checkpoint.get("current_epoch", 0)
    
  def get_model_summary(self) -> Dict[str, Any]:
    """Get model summary information."""
    if self.model is None:
      return {}
      
    total_params = sum(p.numel() for p in self.model.parameters())
    trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    return {
      "total_parameters": total_params,
      "trainable_parameters": trainable_params,
      "device": str(self.device),
      "current_epoch": self.current_epoch,
      "history": self.history,
    }