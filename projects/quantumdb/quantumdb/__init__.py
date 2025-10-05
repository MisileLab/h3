"""
QuantumDB: High-performance vector database with learnable compression.

This package provides a Python interface to train learnable vector compression models
and integrate with Qdrant vector database for production deployments.
"""

from .api import QuantumDB
from .training.model import LearnablePQ
from .training.trainer import Trainer

__version__ = "0.1.0"
__author__ = "QuantumDB Team"

__all__ = [
    "QuantumDB",
    "LearnablePQ",
    "Trainer",
]
