"""
Daily Training Pipeline for Next Action Predictor
Handles incremental training, evaluation, and model export
"""

import logging
import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from pathlib import Path
import pickle

from ..db.postgres import Database
from ..models.moe_model import NextActionMoE, create_model
from ..services.model_exporter import ModelExporter
from ..monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class TrainingConfig:
    """Training configuration"""
    
    def __init__(self):
        self.epochs = 10
        self.batch_size = 64
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.validation_split = 0.2
        self.early_stopping_patience = 5
        self.min_improvement = 0.001
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gradient_clip_norm = 1.0
        self.save_best_only = True
        self.scheduler_patience = 3
        self.scheduler_factor = 0.5


class ActionDataset(Dataset):
    """Dataset for next action prediction"""
    
    def __init__(
        self, 
        features: List[Dict[str, Any]], 
        labels: List[Dict[str, Any]],
        tokenizer=None
    ):
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        
        # Preprocess features
        self.temporal_features = []
        self.url_features = []
        self.behavior_features = []
        
        for feature in features:
            self.temporal_features.append(self._process_temporal_features(feature))
            self.url_features.append(self._process_url_features(feature))
            self.behavior_features.append(self._process_behavior_features(feature))
        
        # Convert to tensors
        self.temporal_features = torch.FloatTensor(self.temporal_features)
        self.url_features = torch.LongTensor(self.url_features)
        self.behavior_features = torch.FloatTensor(self.behavior_features)
        
        # Process labels
        self.tab_labels = []
        self.search_labels = []
        self.scroll_labels = []
        self.router_labels = []
        
        for label in labels:
            self.tab_labels.append(label.get('tab_index', 0))
            self.search_labels.append(label.get('search_tokens', [0] * 32))
            self.scroll_labels.append(label.get('scroll_position', 0.5))
            self.router_labels.append(label.get('action_type', [1, 0, 0]))  # Default to tab
        
        self.tab_labels = torch.LongTensor(self.tab_labels)
        self.search_labels = torch.LongTensor(self.search_labels)
        self.scroll_labels = torch.FloatTensor(self.scroll_labels)
        self.router_labels = torch.FloatTensor(self.router_labels)
    
    def _process_temporal_features(self, feature: Dict[str, Any]) -> List[float]:
        """Process temporal features into fixed-size vector"""
        temporal = [
            feature.get('hour', 0) / 24.0,
            feature.get('minute', 0) / 60.0,
            feature.get('day_of_week', 0) / 7.0,
            float(feature.get('is_weekend', False)),
            feature.get('time_on_page', 0) / 300.0,  # Normalize to 5 minutes
            feature.get('time_since_last_tab_switch', 600) / 600.0,  # Normalize to 10 minutes
            feature.get('time_since_last_search', 600) / 600.0,
            feature.get('time_since_last_navigation', 600) / 600.0,
            feature.get('tab_switch_frequency', 0) / 10.0,
            feature.get('search_frequency', 0) / 5.0
        ]
        
        # Pad or truncate to 10 dimensions
        return temporal[:10] + [0.0] * max(0, 10 - len(temporal))
    
    def _process_url_features(self, feature: Dict[str, Any]) -> List[int]:
        """Process URL features into token sequence"""
        url = feature.get('current_url', '')
        
        # Simple tokenization (in practice, use a proper tokenizer)
        tokens = self._simple_tokenize(url)
        
        # Pad or truncate to fixed length
        max_len = 32
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens = tokens + [0] * (max_len - len(tokens))
        
        return tokens
    
    def _process_behavior_features(self, feature: Dict[str, Any]) -> List[float]:
        """Process behavior features into fixed-size vector"""
        behavior = [
            feature.get('current_tab_index', 0) / 20.0,
            feature.get('total_tabs', 1) / 20.0,
            feature.get('scroll_frequency', 0) / 20.0,
            feature.get('scroll_position', 0.5),
            len(feature.get('recent_urls', [])) / 10.0,
            len(feature.get('recent_domains', [])) / 10.0,
            len(feature.get('recent_searches', [])) / 5.0,
            float(feature.get('is_search_heavy_domain', False)),
            float(feature.get('is_content_heavy_page', False)),
            feature.get('avg_session_duration', 0) / 60.0,  # Normalize to hours
            0.0,  # Placeholder for future features
            0.0,
            0.0,
            0.0,
            0.0
        ]
        
        return behavior[:15]
    
    def _simple_tokenize(self, text: str) -> List[int]:
        """Simple tokenization for URLs"""
        if not text:
            return [0]
        
        # Split URL into components and create simple hash tokens
        components = text.split('/')
        tokens = []
        
        for comp in components:
            if comp:
                # Simple hash to token ID
                token_id = abs(hash(comp)) % 1000 + 1  # Reserve 0 for padding
                tokens.append(token_id)
        
        return tokens[:32]  # Limit length
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.temporal_features[idx],
            self.url_features[idx],
            self.behavior_features[idx],
            self.tab_labels[idx],
            self.search_labels[idx],
            self.scroll_labels[idx],
            self.router_labels[idx]
        )


class DailyTrainer:
    """Daily training pipeline"""
    
    def __init__(
        self, 
        db: Database, 
        model_exporter: ModelExporter,
        metrics: MetricsCollector,
        config: Optional[TrainingConfig] = None
    ):
        self.db = db
        self.model_exporter = model_exporter
        self.metrics = metrics
        self.config = config or TrainingConfig()
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.best_accuracy = 0.0
        self.patience_counter = 0
        
        # Create directories
        self.checkpoints_dir = Path("models/checkpoints")
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    async def run_training(self, force: bool = False) -> Dict[str, Any]:
        """Run the daily training pipeline"""
        try:
            logger.info("Starting daily training pipeline")
            start_time = datetime.utcnow()
            
            # Check if we should run training
            if not force and not await self._should_train():
                logger.info("Skipping training - insufficient new data")
                return {"status": "skipped", "reason": "insufficient_data"}
            
            # Load data
            train_data, val_data = await self._load_training_data()
            if not train_data:
                logger.info("No training data available")
                return {"status": "skipped", "reason": "no_data"}
            
            # Initialize model
            await self._initialize_model()
            
            # Create datasets and dataloaders
            train_dataset = ActionDataset(train_data['features'], train_data['labels'])
            val_dataset = ActionDataset(val_data['features'], val_data['labels'])
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=2
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=2
            )
            
            # Training loop
            training_results = await self._train_model(train_loader, val_loader)
            
            # Evaluate and save if improved
            if training_results['best_val_accuracy'] > self.best_accuracy:
                await self._save_best_model(training_results)
                self.best_accuracy = training_results['best_val_accuracy']
            
            # Export to ONNX
            await self._export_to_onnx()
            
            # Cleanup
            await self._cleanup()
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            results = {
                "status": "completed",
                "duration_seconds": duration,
                "training_results": training_results,
                "model_version": await self._get_next_version()
            }
            
            logger.info(f"Training completed in {duration:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def _should_train(self) -> bool:
        """Check if we should run training based on available data"""
        try:
            # Check if we have enough new data (at least 1000 new events)
            recent_events = await self.db.get_recent_events(minutes=24 * 60)  # Last 24 hours
            
            if len(recent_events) < 1000:
                logger.info(f"Insufficient data: {len(recent_events)} events (need 1000)")
                return False
            
            # Check if we have enough labeled data
            # In practice, this would come from user feedback
            feedback_count = await self._get_feedback_count()
            if feedback_count < 100:
                logger.info(f"Insufficient feedback: {feedback_count} (need 100)")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking training conditions: {e}")
            return False
    
    async def _load_training_data(self) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Load and prepare training data"""
        try:
            # Get events from the last 7 days
            events = await self.db.get_recent_events(minutes=7 * 24 * 60)
            
            if not events:
                return None, None
            
            # Process events into training examples
            training_examples = await self._process_events_to_examples(events)
            
            if not training_examples:
                return None, None
            
            # Split into train/validation
            np.random.shuffle(training_examples)
            split_idx = int(len(training_examples) * (1 - self.config.validation_split))
            
            train_examples = training_examples[:split_idx]
            val_examples = training_examples[split_idx:]
            
            # Separate features and labels
            train_data = {
                'features': [ex['features'] for ex in train_examples],
                'labels': [ex['labels'] for ex in train_examples]
            }
            
            val_data = {
                'features': [ex['features'] for ex in val_examples],
                'labels': [ex['labels'] for ex in val_examples]
            }
            
            logger.info(f"Loaded {len(train_examples)} training and {len(val_examples)} validation examples")
            return train_data, val_data
            
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return None, None
    
    async def _process_events_to_examples(self, events: List) -> List[Dict]:
        """Process raw events into training examples"""
        examples = []
        
        # Group events by sessions or time windows
        # This is a simplified implementation
        for i in range(len(events) - 1):
            current_event = events[i]
            next_event = events[i + 1]
            
            # Create training example
            example = await self._create_training_example(current_event, next_event)
            if example:
                examples.append(example)
        
        return examples
    
    async def _create_training_example(self, current_event: Dict, next_event: Dict) -> Optional[Dict]:
        """Create a single training example from two consecutive events"""
        try:
            # Extract features from current event
            features = await self._extract_features_from_event(current_event)
            
            # Extract labels from next event
            labels = await self._extract_labels_from_event(next_event)
            
            return {
                'features': features,
                'labels': labels
            }
            
        except Exception as e:
            logger.error(f"Failed to create training example: {e}")
            return None
    
    async def _extract_features_from_event(self, event: Dict) -> Dict[str, Any]:
        """Extract features from a single event"""
        # This is a simplified feature extraction
        # In practice, you'd want more sophisticated feature engineering
        
        timestamp = event.get('timestamp', 0)
        dt = datetime.fromtimestamp(timestamp / 1000)
        
        return {
            'hour': dt.hour,
            'minute': dt.minute,
            'day_of_week': dt.weekday(),
            'is_weekend': dt.weekday() >= 5,
            'current_url': event.get('data', {}).get('url', ''),
            'current_tab_index': event.get('data', {}).get('index', 0),
            'total_tabs': 10,  # Would need to track this
            'time_on_page': 0,  # Would need to calculate
            'time_since_last_tab_switch': 0,
            'time_since_last_search': 0,
            'time_since_last_navigation': 0,
            'tab_switch_frequency': 0,
            'search_frequency': 0,
            'scroll_frequency': 0,
            'scroll_position': 0.5,
            'recent_urls': [],
            'recent_domains': [],
            'recent_searches': [],
            'is_search_heavy_domain': False,
            'is_content_heavy_page': False,
            'avg_session_duration': 0
        }
    
    async def _extract_labels_from_event(self, event: Dict) -> Dict[str, Any]:
        """Extract labels from an event"""
        event_type = event.get('type', 'tab_switch')
        
        # Convert event type to router label
        router_labels = {
            'tab_switch': [1, 0, 0],
            'search': [0, 1, 0],
            'scroll': [0, 0, 1]
        }
        
        return {
            'action_type': router_labels.get(event_type, [1, 0, 0]),
            'tab_index': event.get('data', {}).get('index', 0),
            'search_tokens': [0] * 32,  # Would need proper tokenization
            'scroll_position': event.get('data', {}).get('scrollPercentage', 0.5)
        }
    
    async def _initialize_model(self):
        """Initialize or load existing model"""
        try:
            # Try to load existing model
            latest_version = await self.model_exporter.get_latest_version()
            model_path = await self.model_exporter.get_model_file(latest_version['version'])
            
            if model_path and os.path.exists(model_path.replace('.onnx', '.safetensors')):
                # Load existing model
                self.model = torch.load(model_path.replace('.onnx', '.safetensors'), map_location=self.config.device)
                logger.info(f"Loaded existing model from {model_path}")
            else:
                # Create new model
                self.model = create_model()
                logger.info("Created new model")
            
            self.model.to(self.config.device)
            
            # Initialize optimizer and scheduler
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience,
                verbose=True
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    async def _train_model(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """Train the model"""
        try:
            training_results = {
                'train_losses': [],
                'val_losses': [],
                'val_accuracies': [],
                'best_val_accuracy': 0.0,
                'epochs_trained': 0
            }
            
            for epoch in range(self.config.epochs):
                # Training phase
                train_loss = await self._train_epoch(train_loader)
                
                # Validation phase
                val_loss, val_accuracy = await self._validate_epoch(val_loader)
                
                # Update learning rate
                self.scheduler.step(val_accuracy)
                
                # Record metrics
                training_results['train_losses'].append(train_loss)
                training_results['val_losses'].append(val_loss)
                training_results['val_accuracies'].append(val_accuracy)
                training_results['epochs_trained'] = epoch + 1
                
                # Track best accuracy
                if val_accuracy > training_results['best_val_accuracy']:
                    training_results['best_val_accuracy'] = val_accuracy
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Log progress
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs}: "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Acc: {val_accuracy:.4f}"
                )
                
                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            return training_results
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    async def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            temporal_features, url_features, behavior_features, tab_labels, search_labels, scroll_labels, router_labels = batch
            
            # Move to device
            temporal_features = temporal_features.to(self.config.device)
            url_features = url_features.to(self.config.device)
            behavior_features = behavior_features.to(self.config.device)
            tab_labels = tab_labels.to(self.config.device)
            search_labels = search_labels.to(self.config.device)
            scroll_labels = scroll_labels.to(self.config.device)
            router_labels = router_labels.to(self.config.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(temporal_features, url_features, behavior_features)
            
            # Calculate loss
            loss = self._calculate_loss(outputs, tab_labels, search_labels, scroll_labels, router_labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    async def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                temporal_features, url_features, behavior_features, tab_labels, search_labels, scroll_labels, router_labels = batch
                
                # Move to device
                temporal_features = temporal_features.to(self.config.device)
                url_features = url_features.to(self.config.device)
                behavior_features = behavior_features.to(self.config.device)
                tab_labels = tab_labels.to(self.config.device)
                search_labels = search_labels.to(self.config.device)
                scroll_labels = scroll_labels.to(self.config.device)
                router_labels = router_labels.to(self.config.device)
                
                # Forward pass
                outputs = self.model(temporal_features, url_features, behavior_features)
                
                # Calculate loss
                loss = self._calculate_loss(outputs, tab_labels, search_labels, scroll_labels, router_labels)
                
                # Calculate accuracy (simplified - just check router predictions)
                router_pred = outputs['router_weights']
                router_correct = torch.argmax(router_pred, dim=1) == torch.argmax(router_labels, dim=1)
                correct_predictions += router_correct.sum().item()
                total_predictions += router_labels.size(0)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        return avg_loss, accuracy
    
    def _calculate_loss(
        self, 
        outputs: Dict[str, torch.Tensor], 
        tab_labels: torch.Tensor,
        search_labels: torch.Tensor,
        scroll_labels: torch.Tensor,
        router_labels: torch.Tensor
    ) -> torch.Tensor:
        """Calculate multi-task loss"""
        # Router loss
        router_loss = nn.CrossEntropyLoss()(outputs['router_weights'], router_labels)
        
        # Tab expert loss
        tab_loss = nn.CrossEntropyLoss()(outputs['tab_predictions'], tab_labels)
        
        # Search expert loss (simplified)
        search_loss = nn.CrossEntropyLoss()(
            outputs['search_predictions'].view(-1, outputs['search_predictions'].size(-1)),
            search_labels.view(-1)
        )
        
        # Scroll expert loss
        scroll_loss = nn.MSELoss()(outputs['scroll_predictions'].squeeze(), scroll_labels)
        
        # Combine losses with weights
        total_loss = (
            0.3 * router_loss +
            0.4 * tab_loss +
            0.2 * search_loss +
            0.1 * scroll_loss
        )
        
        return total_loss
    
    async def _save_best_model(self, training_results: Dict[str, Any]):
        """Save the best model"""
        try:
            version = await self._get_next_version()
            model_path = self.checkpoints_dir / f"nextpred_model_v{version}.safetensors"
            
            # Save model
            torch.save(self.model.state_dict(), model_path)
            
            # Save training metadata
            metadata = {
                'version': version,
                'training_results': training_results,
                'config': self.config.__dict__,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            metadata_path = self.checkpoints_dir / f"metadata_v{version}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved best model: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    async def _export_to_onnx(self):
        """Export model to ONNX format"""
        try:
            version = await self._get_next_version()
            
            # Create dummy input
            batch_size = 1
            temporal_features = torch.randn(batch_size, 10)
            url_features = torch.randint(0, 1000, (batch_size, 32))
            behavior_features = torch.randn(batch_size, 15)
            
            # Export to ONNX
            await self.model_exporter.export_pytorch_to_onnx(
                str(self.checkpoints_dir / f"nextpred_model_v{version}.safetensors"),
                version,
                input_shape=(batch_size, 64)  # Simplified input shape
            )
            
            logger.info(f"Exported model to ONNX format: version {version}")
            
        except Exception as e:
            logger.error(f"Failed to export to ONNX: {e}")
    
    async def _cleanup(self):
        """Cleanup temporary files and resources"""
        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Remove old checkpoints
            await self.model_exporter.cleanup_old_models()
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    async def _get_feedback_count(self) -> int:
        """Get count of recent feedback"""
        # This would query the feedback table
        # For now, return a mock value
        return 150
    
    async def _get_next_version(self) -> str:
        """Get next model version"""
        latest = await self.model_exporter.get_latest_version()
        current_version = latest.get('version', '1.0.0')
        
        # Simple version increment
        parts = current_version.split('.')
        patch = int(parts[2]) + 1
        return f"{parts[0]}.{parts[1]}.{patch}"


async def run_training(force: bool = False):
    """Main entry point for training pipeline"""
    try:
        # Initialize components
        db = Database()
        await db.initialize()
        
        model_exporter = ModelExporter(db)
        metrics = MetricsCollector()
        
        # Create trainer
        trainer = DailyTrainer(db, model_exporter, metrics)
        
        # Run training
        results = await trainer.run_training(force=force)
        
        # Log results
        logger.info(f"Training results: {results}")
        
        # Close connections
        await db.close()
        
        return results
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    # Run training
    asyncio.run(run_training(force=True))