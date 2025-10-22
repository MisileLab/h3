"""
Model Exporter Service for Next Action Predictor
Handles model versioning, ONNX export, and file management
"""

import logging
from typing import List, Dict, Any, Optional
import os
import hashlib
import json
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from ..db.postgres import Database

logger = logging.getLogger(__name__)


class ModelExporter:
    """Service for exporting and managing ML models"""

    def __init__(self, db: Database):
        self.db = db
        self.models_dir = Path("models")
        self.onnx_dir = self.models_dir / "onnx"
        self.checkpoints_dir = self.models_dir / "checkpoints"
        
        # Create directories if they don't exist
        self.models_dir.mkdir(exist_ok=True)
        self.onnx_dir.mkdir(exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)

    async def get_latest_version(self) -> Dict[str, Any]:
        """Get latest model version information"""
        try:
            model_version = await self.db.get_model_version()
            
            if not model_version:
                # Return default version info
                return {
                    "version": "1.0.0",
                    "timestamp": datetime.utcnow(),
                    "accuracy_metrics": {
                        "top1_accuracy": 0.0,
                        "top3_accuracy": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1_score": 0.0
                    },
                    "file_size": None,
                    "checksum": None
                }
            
            return {
                "version": model_version.version,
                "timestamp": model_version.created_at,
                "accuracy_metrics": model_version.accuracy_metrics or {},
                "file_size": model_version.file_size,
                "checksum": model_version.checksum
            }

        except Exception as e:
            logger.error(f"Failed to get latest model version: {e}")
            raise

    async def get_model_file(self, version: str) -> Optional[str]:
        """Get file path for a specific model version"""
        try:
            model_version = await self.db.get_model_version(version)
            
            if not model_version or not model_version.file_path:
                return None
            
            # Check if file exists
            if os.path.exists(model_version.file_path):
                return model_version.file_path
            
            # Try to find the file in models directory
            model_filename = f"nextpred_model_v{version}.onnx"
            model_path = self.onnx_dir / model_filename
            
            if model_path.exists():
                # Update database with correct path
                await self._update_model_path(version, str(model_path))
                return str(model_path)
            
            return None

        except Exception as e:
            logger.error(f"Failed to get model file for version {version}: {e}")
            return None

    async def store_model(
        self, 
        file_data, 
        version: str,
        accuracy_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Store a new model version"""
        try:
            # Generate filename
            model_filename = f"nextpred_model_v{version}.onnx"
            model_path = self.onnx_dir / model_filename
            
            # Save file to disk
            with open(model_path, 'wb') as f:
                if hasattr(file_data, 'read'):
                    # File-like object
                    shutil.copyfileobj(file_data, f)
                else:
                    # Raw bytes
                    f.write(file_data)
            
            # Calculate checksum
            checksum = await self._calculate_file_checksum(model_path)
            file_size = model_path.stat().st_size
            
            # Store in database
            model_data = {
                "version": version,
                "file_path": str(model_path),
                "file_size": file_size,
                "checksum": checksum,
                "accuracy_metrics": accuracy_metrics or {}
            }
            
            model_id = await self.db.store_model_version(model_data)
            
            logger.info(f"Stored model version {version} with ID {model_id}")
            
            return {
                "id": model_id,
                "version": version,
                "file_path": str(model_path),
                "file_size": file_size,
                "checksum": checksum,
                "accuracy_metrics": accuracy_metrics
            }

        except Exception as e:
            logger.error(f"Failed to store model version {version}: {e}")
            raise

    async def export_pytorch_to_onnx(
        self, 
        pytorch_model_path: str,
        version: str,
        input_shape: tuple = (1, 64),
        opset_version: int = 11
    ) -> Dict[str, Any]:
        """Export PyTorch model to ONNX format"""
        try:
            import torch
            import torch.onnx
            
            # Load PyTorch model
            model = torch.load(pytorch_model_path, map_location='cpu')
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(input_shape)
            
            # Generate ONNX filename
            onnx_filename = f"nextpred_model_v{version}.onnx"
            onnx_path = self.onnx_dir / onnx_filename
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # Calculate checksum and file size
            checksum = await self._calculate_file_checksum(onnx_path)
            file_size = onnx_path.stat().st_size
            
            # Store in database
            model_data = {
                "version": version,
                "file_path": str(onnx_path),
                "file_size": file_size,
                "checksum": checksum,
                "accuracy_metrics": {}
            }
            
            model_id = await self.db.store_model_version(model_data)
            
            logger.info(f"Exported PyTorch model to ONNX: {onnx_path}")
            
            return {
                "id": model_id,
                "version": version,
                "onnx_path": str(onnx_path),
                "file_size": file_size,
                "checksum": checksum
            }

        except ImportError:
            logger.error("PyTorch not available for ONNX export")
            raise
        except Exception as e:
            logger.error(f"Failed to export PyTorch model to ONNX: {e}")
            raise

    async def validate_model(self, model_path: str) -> Dict[str, Any]:
        """Validate ONNX model file"""
        try:
            import onnx
            import onnxruntime as ort
            
            # Load and validate ONNX model
            model = onnx.load(model_path)
            onnx.checker.check_model(model)
            
            # Create inference session to test
            session = ort.InferenceSession(model_path)
            
            # Get input/output info
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            
            # Test inference with dummy data
            import numpy as np
            dummy_input = np.random.random(
                (input_info.shape[0] if input_info.shape[0] != 'None' else 1, 
                 input_info.shape[1] if len(input_info.shape) > 1 else 64)
            ).astype(np.float32)
            
            outputs = session.run(None, {input_info.name: dummy_input})
            
            validation_result = {
                "valid": True,
                "input_name": input_info.name,
                "input_shape": input_info.shape,
                "input_type": str(input_info.type),
                "output_name": output_info.name,
                "output_shape": outputs[0].shape,
                "output_type": str(outputs[0].dtype),
                "providers": session.get_providers()
            }
            
            logger.info(f"Model validation successful for {model_path}")
            return validation_result

        except ImportError:
            logger.error("ONNX libraries not available for validation")
            return {"valid": False, "error": "ONNX libraries not available"}
        except Exception as e:
            logger.error(f"Model validation failed for {model_path}: {e}")
            return {"valid": False, "error": str(e)}

    async def create_model_package(
        self, 
        version: str,
        include_metadata: bool = True
    ) -> str:
        """Create a packaged model archive"""
        try:
            import zipfile
            
            model_path = await self.get_model_file(version)
            if not model_path:
                raise ValueError(f"Model version {version} not found")
            
            # Create package filename
            package_filename = f"nextpred_model_v{version}_package.zip"
            package_path = self.models_dir / package_filename
            
            with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add ONNX model
                zipf.write(model_path, os.path.basename(model_path))
                
                if include_metadata:
                    # Add metadata file
                    metadata = await self.get_latest_version()
                    metadata_path = self.models_dir / f"metadata_v{version}.json"
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2, default=str)
                    
                    zipf.write(metadata_path, "metadata.json")
                    
                    # Clean up metadata file
                    os.remove(metadata_path)
            
            logger.info(f"Created model package: {package_path}")
            return str(package_path)

        except Exception as e:
            logger.error(f"Failed to create model package for version {version}: {e}")
            raise

    async def cleanup_old_models(self, keep_versions: int = 5) -> int:
        """Clean up old model versions, keeping only the most recent ones"""
        try:
            # Get all model versions from database
            # This would require a new method in the database class
            # For now, implement a simple file-based cleanup
            
            onnx_files = list(self.onnx_dir.glob("*.onnx"))
            onnx_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            deleted_count = 0
            for old_file in onnx_files[keep_versions:]:
                try:
                    os.remove(old_file)
                    deleted_count += 1
                    logger.info(f"Deleted old model file: {old_file}")
                except Exception as e:
                    logger.error(f"Failed to delete {old_file}: {e}")
            
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old models: {e}")
            return 0

    async def get_model_info(self, version: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model version"""
        try:
            model_version = await self.db.get_model_version(version)
            
            if not model_version:
                return None
            
            model_path = await self.get_model_file(version)
            validation_result = None
            
            if model_path and os.path.exists(model_path):
                validation_result = await self.validate_model(model_path)
            
            return {
                "version": model_version.version,
                "created_at": model_version.created_at,
                "file_path": model_version.file_path,
                "file_size": model_version.file_size,
                "checksum": model_version.checksum,
                "accuracy_metrics": model_version.accuracy_metrics,
                "is_active": model_version.is_active,
                "download_count": model_version.download_count,
                "validation": validation_result
            }

        except Exception as e:
            logger.error(f"Failed to get model info for version {version}: {e}")
            return None

    async def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of a file"""
        try:
            sha256_hash = hashlib.sha256()
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            return sha256_hash.hexdigest()

        except Exception as e:
            logger.error(f"Failed to calculate checksum for {file_path}: {e}")
            return ""

    async def _update_model_path(self, version: str, file_path: str):
        """Update model file path in database"""
        try:
            # This would require updating the database class
            # For now, just log the update
            logger.info(f"Would update model {version} path to {file_path}")

        except Exception as e:
            logger.error(f"Failed to update model path: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Get model exporter status"""
        try:
            # Count model files
            onnx_files = list(self.onnx_dir.glob("*.onnx"))
            checkpoint_files = list(self.checkpoints_dir.glob("*.safetensors"))
            
            # Get latest version info
            latest_version = await self.get_latest_version()
            
            return {
                "status": "healthy",
                "models_directory": str(self.models_dir),
                "onnx_files_count": len(onnx_files),
                "checkpoint_files_count": len(checkpoint_files),
                "latest_version": latest_version.get("version"),
                "disk_usage": {
                    "models_dir": self._get_directory_size(self.models_dir),
                    "onnx_dir": self._get_directory_size(self.onnx_dir),
                    "checkpoints_dir": self._get_directory_size(self.checkpoints_dir)
                }
            }

        except Exception as e:
            logger.error(f"Failed to get model exporter status: {e}")
            return {"status": "unhealthy", "error": str(e)}

    def _get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes"""
        try:
            total_size = 0
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except Exception:
            return 0