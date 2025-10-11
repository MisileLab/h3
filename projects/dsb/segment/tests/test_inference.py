"""
Tests for the Inference Module

This module contains tests for the inference functionality including
single predictions, batch predictions, and edge cases.
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
import sys
sys.path.append(str(project_root))

try:
    from src.inference import JailbreakDetector, InferenceResult, predict, predict_batch
except ImportError:
    pytest.skip("Required dependencies not installed", allow_module_level=True)


class TestInferenceResult:
    """Test class for InferenceResult dataclass"""
    
    def test_inference_result_creation(self):
        """Test InferenceResult creation"""
        result = InferenceResult(
            text="test text",
            label="safe",
            prediction=0,
            probability=0.1,
            confidence=0.9,
            processing_time=0.1
        )
        
        assert result.text == "test text"
        assert result.label == "safe"
        assert result.prediction == 0
        assert result.probability == 0.1
        assert result.confidence == 0.9
        assert result.processing_time == 0.1


class TestJailbreakDetector:
    """Test class for JailbreakDetector"""
    
    @pytest.fixture
    def mock_model(self):
        """Mock model for testing"""
        model = Mock()
        model.predict.return_value = {
            'predictions': [0],
            'labels': ['safe'],
            'probabilities': [[0.9, 0.1]],
            'unsafe_probabilities': [0.1]
        }
        return model
    
    @pytest.fixture
    def temp_dir(self):
        """Temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_detector_initialization(self, temp_dir):
        """Test detector initialization"""
        # Mock the model loading
        with patch('src.inference.SegmentModel.load_model') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            detector = JailbreakDetector(
                model_path="dummy_path",
                threshold=0.7
            )
            
            assert detector.model_path == "dummy_path"
            assert detector.threshold == 0.7
            assert detector.model == mock_model
    
    def test_predict_single(self, mock_model, temp_dir):
        """Test single text prediction"""
        with patch('src.inference.SegmentModel.load_model', return_value=mock_model):
            detector = JailbreakDetector(model_path="dummy_path")
            
            result = detector.predict_single("test text")
            
            assert isinstance(result, InferenceResult)
            assert result.text == "test text"
            assert result.label in ['safe', 'unsafe']
            assert result.prediction in [0, 1]
            assert 0 <= result.probability <= 1
            assert 0 <= result.confidence <= 1
            assert result.processing_time >= 0
    
    def test_predict_batch(self, mock_model, temp_dir):
        """Test batch text prediction"""
        # Mock batch prediction
        mock_model.predict.return_value = {
            'predictions': [0, 1, 0],
            'labels': ['safe', 'unsafe', 'safe'],
            'probabilities': [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2]],
            'unsafe_probabilities': [0.1, 0.8, 0.2]
        }
        
        with patch('src.inference.SegmentModel.load_model', return_value=mock_model):
            detector = JailbreakDetector(model_path="dummy_path")
            
            texts = ["text1", "text2", "text3"]
            results = detector.predict_batch(texts)
            
            assert len(results) == 3
            for i, result in enumerate(results):
                assert isinstance(result, InferenceResult)
                assert result.text == texts[i]
                assert result.label in ['safe', 'unsafe']
                assert result.prediction in [0, 1]
    
    def test_is_safe(self, mock_model, temp_dir):
        """Test is_safe method"""
        with patch('src.inference.SegmentModel.load_model', return_value=mock_model):
            detector = JailbreakDetector(model_path="dummy_path")
            
            # Test safe prediction
            assert detector.is_safe("safe text") is True
            
            # Test unsafe prediction
            mock_model.predict.return_value = {
                'predictions': [1],
                'labels': ['unsafe'],
                'probabilities': [[0.1, 0.9]],
                'unsafe_probabilities': [0.9]
            }
            assert detector.is_safe("unsafe text") is False
    
    def test_get_unsafe_probability(self, mock_model, temp_dir):
        """Test get_unsafe_probability method"""
        with patch('src.inference.SegmentModel.load_model', return_value=mock_model):
            detector = JailbreakDetector(model_path="dummy_path")
            
            prob = detector.get_unsafe_probability("test text")
            assert prob == 0.1
            assert 0 <= prob <= 1
    
    def test_analyze_prompt(self, mock_model, temp_dir):
        """Test analyze_prompt method"""
        with patch('src.inference.SegmentModel.load_model', return_value=mock_model):
            detector = JailbreakDetector(model_path="dummy_path")
            
            analysis = detector.analyze_prompt("test text", return_details=True)
            
            assert 'text' in analysis
            assert 'is_safe' in analysis
            assert 'is_unsafe' in analysis
            assert 'label' in analysis
            assert 'unsafe_probability' in analysis
            assert 'confidence' in analysis
            assert 'processing_time_ms' in analysis
            assert 'threshold_used' in analysis
            assert 'model_path' in analysis
            assert 'max_length' in analysis
            
            assert analysis['text'] == "test text"
            assert analysis['is_safe'] is True
            assert analysis['is_unsafe'] is False
    
    def test_filter_safe_texts(self, mock_model, temp_dir):
        """Test filter_safe_texts method"""
        # Mock predictions for multiple texts
        mock_model.predict.return_value = {
            'predictions': [0, 1, 0, 1],
            'labels': ['safe', 'unsafe', 'safe', 'unsafe'],
            'probabilities': [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]],
            'unsafe_probabilities': [0.1, 0.8, 0.2, 0.9]
        }
        
        with patch('src.inference.SegmentModel.load_model', return_value=mock_model):
            detector = JailbreakDetector(model_path="dummy_path")
            
            texts = ["safe1", "unsafe1", "safe2", "unsafe2"]
            safe_texts = detector.filter_safe_texts(texts)
            
            assert len(safe_texts) == 2
            assert "safe1" in safe_texts
            assert "safe2" in safe_texts
            assert "unsafe1" not in safe_texts
            assert "unsafe2" not in safe_texts
    
    def test_filter_unsafe_texts(self, mock_model, temp_dir):
        """Test filter_unsafe_texts method"""
        # Mock predictions for multiple texts
        mock_model.predict.return_value = {
            'predictions': [0, 1, 0, 1],
            'labels': ['safe', 'unsafe', 'safe', 'unsafe'],
            'probabilities': [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]],
            'unsafe_probabilities': [0.1, 0.8, 0.2, 0.9]
        }
        
        with patch('src.inference.SegmentModel.load_model', return_value=mock_model):
            detector = JailbreakDetector(model_path="dummy_path")
            
            texts = ["safe1", "unsafe1", "safe2", "unsafe2"]
            unsafe_texts = detector.filter_unsafe_texts(texts)
            
            assert len(unsafe_texts) == 2
            assert "unsafe1" in unsafe_texts
            assert "unsafe2" in unsafe_texts
            assert "safe1" not in unsafe_texts
            assert "safe2" not in unsafe_texts
    
    def test_get_statistics(self, mock_model, temp_dir):
        """Test get_statistics method"""
        # Mock predictions for multiple texts
        mock_model.predict.return_value = {
            'predictions': [0, 1, 0, 1],
            'labels': ['safe', 'unsafe', 'safe', 'unsafe'],
            'probabilities': [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]],
            'unsafe_probabilities': [0.1, 0.8, 0.2, 0.9]
        }
        
        with patch('src.inference.SegmentModel.load_model', return_value=mock_model):
            detector = JailbreakDetector(model_path="dummy_path")
            
            texts = ["text1", "text2", "text3", "text4"]
            stats = detector.get_statistics(texts)
            
            assert 'total_texts' in stats
            assert 'safe_count' in stats
            assert 'unsafe_count' in stats
            assert 'safe_percentage' in stats
            assert 'unsafe_percentage' in stats
            assert 'average_unsafe_probability' in stats
            assert 'average_confidence' in stats
            assert 'average_processing_time_ms' in stats
            
            assert stats['total_texts'] == 4
            assert stats['safe_count'] == 2
            assert stats['unsafe_count'] == 2
            assert stats['safe_percentage'] == 50.0
            assert stats['unsafe_percentage'] == 50.0
    
    def test_threshold_application(self, temp_dir):
        """Test threshold application in predictions"""
        # Test with different thresholds
        mock_model = Mock()
        mock_model.predict.return_value = {
            'predictions': [0],
            'labels': ['safe'],
            'probabilities': [[0.6, 0.4]],
            'unsafe_probabilities': [0.4]
        }
        
        with patch('src.inference.SegmentModel.load_model', return_value=mock_model):
            # With threshold 0.5, should be safe
            detector1 = JailbreakDetector(model_path="dummy_path", threshold=0.5)
            result1 = detector1.predict_single("test")
            assert result1.label == 'safe'
            
            # With threshold 0.3, should be unsafe
            detector2 = JailbreakDetector(model_path="dummy_path", threshold=0.3)
            result2 = detector2.predict_single("test")
            assert result2.label == 'unsafe'


class TestStandaloneFunctions:
    """Test standalone inference functions"""
    
    def test_predict_function(self, temp_dir):
        """Test standalone predict function"""
        with patch('src.inference.SegmentModel.load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = {
                'predictions': [0],
                'labels': ['safe'],
                'probabilities': [[0.9, 0.1]],
                'unsafe_probabilities': [0.1]
            }
            mock_load.return_value = mock_model
            
            result = predict("test text", "dummy_path")
            
            assert 'text' in result
            assert 'label' in result
            assert 'is_safe' in result
            assert 'unsafe_probability' in result
            assert 'confidence' in result
            
            assert result['text'] == "test text"
            assert result['label'] == 'safe'
            assert result['is_safe'] is True
    
    def test_predict_batch_function(self, temp_dir):
        """Test standalone predict_batch function"""
        with patch('src.inference.SegmentModel.load_model') as mock_load:
            mock_model = Mock()
            mock_model.predict.return_value = {
                'predictions': [0, 1],
                'labels': ['safe', 'unsafe'],
                'probabilities': [[0.9, 0.1], [0.2, 0.8]],
                'unsafe_probabilities': [0.1, 0.8]
            }
            mock_load.return_value = mock_model
            
            texts = ["text1", "text2"]
            results = predict_batch(texts, "dummy_path")
            
            assert len(results) == 2
            for result in results:
                assert 'text' in result
                assert 'label' in result
                assert 'is_safe' in result
                assert 'unsafe_probability' in result
                assert 'confidence' in result


if __name__ == "__main__":
    pytest.main([__file__])