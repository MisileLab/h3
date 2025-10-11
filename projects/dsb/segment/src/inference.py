"""
Inference Module for ELECTRA-based Jailbreak Detection Model

This module provides easy-to-use inference functions for detecting jailbreak attempts
in prompts, with support for batch processing and real-time detection.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Union, Optional
import time
from dataclasses import dataclass

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import SegmentModel
from src.utils import setup_logging, get_device

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Data class for inference results"""
    text: str
    label: str  # 'safe' or 'unsafe'
    prediction: int  # 0 or 1
    probability: float  # Probability of being unsafe
    confidence: float  # Confidence in the prediction
    processing_time: float  # Time taken for inference in seconds


class JailbreakDetector:
    """
    High-level interface for jailbreak detection
    
    This class provides a simple API for detecting jailbreak attempts in prompts.
    """
    
    def __init__(
        self, 
        model_path: str, 
        config_path: Optional[str] = None,
        threshold: float = 0.5,
        device: Optional[str] = None
    ):
        """
        Initialize the jailbreak detector
        
        Args:
            model_path: Path to trained model
            config_path: Path to configuration file (optional)
            threshold: Classification threshold for unsafe detection
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.model_path = model_path
        self.threshold = threshold
        
        # Load configuration if provided
        config = {}
        if config_path and os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        # Initialize model
        if device is None:
            device = get_device(config)
            
        self.model = SegmentModel.load_model(model_path, device=device)
        self.max_length = config.get('data', {}).get('max_length', 512)
        
        logger.info(f"JailbreakDetector initialized with model: {model_path}")
        logger.info(f"Classification threshold: {threshold}")
        
    def predict_single(self, text: str) -> InferenceResult:
        """
        Predict jailbreak status for a single text
        
        Args:
            text: Input text to analyze
            
        Returns:
            InferenceResult with prediction details
        """
        start_time = time.time()
        
        # Make prediction
        results = self.model.predict([text], max_length=self.max_length, return_probabilities=True)
        
        processing_time = time.time() - start_time
        
        # Extract results
        prediction = results['predictions'][0]
        label = results['labels'][0]
        unsafe_prob = results['unsafe_probabilities'][0]
        
        # Calculate confidence (max probability)
        probabilities = results['probabilities'][0]
        confidence = max(probabilities)
        
        # Apply threshold
        if unsafe_prob >= self.threshold:
            final_label = 'unsafe'
            final_prediction = 1
        else:
            final_label = 'safe'
            final_prediction = 0
        
        return InferenceResult(
            text=text,
            label=final_label,
            prediction=final_prediction,
            probability=unsafe_prob,
            confidence=confidence,
            processing_time=processing_time
        )
    
    def predict_batch(self, texts: List[str]) -> List[InferenceResult]:
        """
        Predict jailbreak status for multiple texts
        
        Args:
            texts: List of input texts to analyze
            
        Returns:
            List of InferenceResult objects
        """
        start_time = time.time()
        
        # Make predictions
        results = self.model.predict(texts, max_length=self.max_length, return_probabilities=True)
        
        processing_time = time.time() - start_time
        
        # Process results
        inference_results = []
        for i, text in enumerate(texts):
            prediction = results['predictions'][i]
            label = results['labels'][i]
            unsafe_prob = results['unsafe_probabilities'][i]
            probabilities = results['probabilities'][i]
            confidence = max(probabilities)
            
            # Apply threshold
            if unsafe_prob >= self.threshold:
                final_label = 'unsafe'
                final_prediction = 1
            else:
                final_label = 'safe'
                final_prediction = 0
            
            inference_results.append(InferenceResult(
                text=text,
                label=final_label,
                prediction=final_prediction,
                probability=unsafe_prob,
                confidence=confidence,
                processing_time=processing_time / len(texts)  # Average time per text
            ))
        
        return inference_results
    
    def is_safe(self, text: str) -> bool:
        """
        Quick check if text is safe
        
        Args:
            text: Input text to analyze
            
        Returns:
            True if text is safe, False if unsafe
        """
        result = self.predict_single(text)
        return result.label == 'safe'
    
    def get_unsafe_probability(self, text: str) -> float:
        """
        Get probability that text is unsafe
        
        Args:
            text: Input text to analyze
            
        Returns:
            Probability (0.0 to 1.0) that text is unsafe
        """
        result = self.predict_single(text)
        return result.probability
    
    def analyze_prompt(self, text: str, return_details: bool = True) -> Dict[str, Any]:
        """
        Comprehensive analysis of a prompt
        
        Args:
            text: Input text to analyze
            return_details: Whether to return detailed analysis
            
        Returns:
            Dictionary with analysis results
        """
        result = self.predict_single(text)
        
        analysis = {
            'text': text,
            'is_safe': result.label == 'safe',
            'is_unsafe': result.label == 'unsafe',
            'label': result.label,
            'unsafe_probability': result.probability,
            'confidence': result.confidence,
            'processing_time_ms': result.processing_time * 1000,
            'threshold_used': self.threshold
        }
        
        if return_details:
            analysis.update({
                'raw_prediction': result.prediction,
                'model_path': self.model_path,
                'max_length': self.max_length
            })
        
        return analysis
    
    def filter_safe_texts(self, texts: List[str]) -> List[str]:
        """
        Filter out unsafe texts, returning only safe ones
        
        Args:
            texts: List of input texts
            
        Returns:
            List of texts that are classified as safe
        """
        results = self.predict_batch(texts)
        safe_texts = [result.text for result in results if result.label == 'safe']
        return safe_texts
    
    def filter_unsafe_texts(self, texts: List[str]) -> List[str]:
        """
        Filter out safe texts, returning only unsafe ones
        
        Args:
            texts: List of input texts
            
        Returns:
            List of texts that are classified as unsafe
        """
        results = self.predict_batch(texts)
        unsafe_texts = [result.text for result in results if result.label == 'unsafe']
        return unsafe_texts
    
    def get_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """
        Get statistics for a batch of texts
        
        Args:
            texts: List of input texts
            
        Returns:
            Dictionary with statistics
        """
        results = self.predict_batch(texts)
        
        total_texts = len(results)
        safe_count = sum(1 for r in results if r.label == 'safe')
        unsafe_count = total_texts - safe_count
        
        avg_unsafe_prob = sum(r.probability for r in results) / total_texts
        avg_confidence = sum(r.confidence for r in results) / total_texts
        avg_processing_time = sum(r.processing_time for r in results) / total_texts
        
        return {
            'total_texts': total_texts,
            'safe_count': safe_count,
            'unsafe_count': unsafe_count,
            'safe_percentage': (safe_count / total_texts) * 100,
            'unsafe_percentage': (unsafe_count / total_texts) * 100,
            'average_unsafe_probability': avg_unsafe_prob,
            'average_confidence': avg_confidence,
            'average_processing_time_ms': avg_processing_time * 1000
        }


def predict(text: str, model_path: str, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Simple prediction function for single text
    
    Args:
        text: Input text to analyze
        model_path: Path to trained model
        threshold: Classification threshold
        
    Returns:
        Dictionary with prediction results
    """
    detector = JailbreakDetector(model_path, threshold=threshold)
    result = detector.predict_single(text)
    
    return {
        'text': text,
        'label': result.label,
        'is_safe': result.label == 'safe',
        'unsafe_probability': result.probability,
        'confidence': result.confidence
    }


def predict_batch(texts: List[str], model_path: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Simple prediction function for multiple texts
    
    Args:
        texts: List of input texts to analyze
        model_path: Path to trained model
        threshold: Classification threshold
        
    Returns:
        List of dictionaries with prediction results
    """
    detector = JailbreakDetector(model_path, threshold=threshold)
    results = detector.predict_batch(texts)
    
    return [
        {
            'text': result.text,
            'label': result.label,
            'is_safe': result.label == 'safe',
            'unsafe_probability': result.probability,
            'confidence': result.confidence
        }
        for result in results
    ]


def main():
    """Command line interface for inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Jailbreak detection inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--text", type=str, help="Single text to analyze")
    parser.add_argument("--file", type=str, help="File containing texts to analyze (one per line)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Initialize detector
    detector = JailbreakDetector(args.model_path, threshold=args.threshold)
    
    if args.text:
        # Single text prediction
        result = detector.predict_single(args.text)
        print(f"Text: {args.text}")
        print(f"Label: {result.label}")
        print(f"Unsafe Probability: {result.probability:.4f}")
        print(f"Confidence: {result.confidence:.4f}")
        print(f"Processing Time: {result.processing_time*1000:.2f}ms")
        
    elif args.file:
        # Batch prediction from file
        with open(args.file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = detector.predict_batch(texts)
        
        # Print results
        for result in results:
            print(f"Text: {result.text[:100]}{'...' if len(result.text) > 100 else ''}")
            print(f"Label: {result.label} (prob: {result.probability:.4f})")
            print("-" * 50)
        
        # Save results if output file specified
        if args.output:
            output_data = [
                {
                    'text': result.text,
                    'label': result.label,
                    'unsafe_probability': result.probability,
                    'confidence': result.confidence
                }
                for result in results
            ]
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Results saved to: {args.output}")
    
    else:
        print("Please provide either --text or --file argument")
        parser.print_help()


if __name__ == "__main__":
    main()