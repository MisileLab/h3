"""
Utility Functions for Segment Project

This module provides common utility functions including logging setup,
device management, configuration handling, and other helper functions.
"""

import os
import sys
import json
import yaml
import logging
import time
import torch
from typing import Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import logging.handlers


def setup_logging(log_dir: Union[str, Path], level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory to save log files
        level: Logging level
        
    Returns:
        Configured logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # File handler with rotation
    log_file = log_dir / f"segment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info(f"Logging setup complete. Log file: {log_file}")
    
    return logger


def get_device(config: Optional[Dict[str, Any]] = None) -> str:
    """
    Get the appropriate device for computation
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if config is None:
        config = {}
    
    device_config = config.get('hardware', {}).get('device', 'auto')
    
    if device_config == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = 'cpu'
            logger.info("CUDA not available, using CPU")
    else:
        device = device_config
        logger.info(f"Using specified device: {device}")
    
    return device


def save_config(config: Dict[str, Any], output_dir: Union[str, Path]):
    """
    Save configuration to file
    
    Args:
        config: Configuration dictionary
        output_dir: Directory to save configuration
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = output_dir / 'config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    # Also save as JSON for easy reading
    json_file = output_dir / 'config.json'
    with open(json_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration saved to {config_file}")


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    
    return config


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed
    """
    import random
    
    random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def get_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Get model size information
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model size information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'size_mb': size_mb,
        'trainable_percentage': (trainable_params / total_params) * 100 if total_params > 0 else 0
    }


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def create_directory_structure(base_dir: Union[str, Path], structure: Dict[str, Any]):
    """
    Create directory structure from dictionary
    
    Args:
        base_dir: Base directory
        structure: Dictionary representing directory structure
    """
    base_dir = Path(base_dir)
    
    for name, content in structure.items():
        path = base_dir / name
        
        if isinstance(content, dict):
            # Create subdirectory
            path.mkdir(parents=True, exist_ok=True)
            create_directory_structure(path, content)
        else:
            # Create file
            path.parent.mkdir(parents=True, exist_ok=True)
            if content is not None:
                with open(path, 'w') as f:
                    f.write(content)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises exception if invalid
    """
    required_sections = ['training', 'data', 'model']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate training config
    training = config['training']
    required_training_keys = ['output_dir', 'num_epochs', 'batch_size', 'learning_rate']
    for key in required_training_keys:
        if key not in training:
            raise ValueError(f"Missing required training configuration: {key}")
    
    # Validate data config
    data = config['data']
    required_data_keys = ['max_length', 'seed']
    for key in required_data_keys:
        if key not in data:
            raise ValueError(f"Missing required data configuration: {key}")
    
    # Validate model config
    model = config['model']
    required_model_keys = ['name', 'num_labels']
    for key in required_model_keys:
        if key not in model:
            raise ValueError(f"Missing required model configuration: {key}")
    
    logger.info("Configuration validation passed")
    return True


def get_git_info() -> Dict[str, str]:
    """
    Get git repository information
    
    Returns:
        Dictionary with git information
    """
    try:
        import subprocess
        
        # Get git commit hash
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        # Get git branch
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        
        # Get git remote URL
        try:
            remote_url = subprocess.check_output(
                ['git', 'config', '--get', 'remote.origin.url'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
        except:
            remote_url = "unknown"
        
        return {
            'commit_hash': commit_hash,
            'branch': branch,
            'remote_url': remote_url
        }
    
    except Exception as e:
        logger.warning(f"Could not get git information: {e}")
        return {
            'commit_hash': 'unknown',
            'branch': 'unknown',
            'remote_url': 'unknown'
        }


def log_system_info():
    """Log system information for debugging"""
    import platform
    import psutil
    
    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python version: {platform.python_version()}")
    logger.info(f"PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # CPU and memory info
    logger.info(f"CPU cores: {psutil.cpu_count()}")
    memory = psutil.virtual_memory()
    logger.info(f"Total memory: {memory.total / 1024**3:.1f} GB")
    logger.info(f"Available memory: {memory.available / 1024**3:.1f} GB")
    
    # Git info
    git_info = get_git_info()
    logger.info(f"Git commit: {git_info['commit_hash']}")
    logger.info(f"Git branch: {git_info['branch']}")
    
    logger.info("=== End System Information ===")


class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"Starting {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if self.start_time is not None:
            elapsed = self.end_time - self.start_time
            logger.info(f"Completed {self.name} in {format_time(elapsed)}")


# Initialize module logger
logger = logging.getLogger(__name__)