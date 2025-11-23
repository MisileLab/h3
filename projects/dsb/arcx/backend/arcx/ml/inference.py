"""Real-time inference engine

Manages screen capture, encoding, and EV prediction in real-time.
"""

import logging
import time
import threading
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import torch
import numpy as np

from arcx.ml.model import EVModel
from arcx.capture import create_capture, CombinedRingBuffer
from arcx.device import device_manager
from arcx.config import config

logger = logging.getLogger(__name__)


@dataclass
class EVPrediction:
    """Container for EV prediction results"""

    ev_stay: float
    ev_extract: float
    delta_ev: float
    recommendation: str  # "stay", "extract", "neutral"
    confidence: float
    timestamp: float
    quantile: float
    inference_time_ms: float

    # Metadata
    buffer_size: int = 0
    is_buffer_full: bool = False


class InferenceEngine:
    """
    Real-time inference engine for EV prediction.

    Manages:
    - Screen capture
    - Frame encoding
    - Ring buffer maintenance
    - Periodic EV prediction
    - Thread-safe state management
    """

    def __init__(
        self,
        model: EVModel,
        capture_fps: int = 8,
        inference_interval: float = 0.5,
        sequence_length: int = 32,
    ):
        """
        Args:
            model: EVModel instance
            capture_fps: Screen capture FPS
            inference_interval: Time between EV predictions (seconds)
            sequence_length: Number of frames in sequence for Q-net
        """
        self.model = model
        self.model.eval()  # Set to eval mode
        self.device = device_manager.device

        self.capture_fps = capture_fps
        self.inference_interval = inference_interval
        self.sequence_length = sequence_length

        # Capture and buffer
        self.capture = create_capture(
            target_fps=capture_fps,
            target_width=config.capture.width,
            target_height=config.capture.height,
        )

        self.buffer = CombinedRingBuffer(
            capacity=config.capture.buffer_size,
            frame_shape=(config.capture.height, config.capture.width, 3),
            latent_dim=model.latent_dim,
        )

        # State
        self.is_running = False
        self.last_prediction: Optional[EVPrediction] = None
        self.capture_thread: Optional[threading.Thread] = None
        self.inference_thread: Optional[threading.Thread] = None

        # Stats
        self.total_frames = 0
        self.total_inferences = 0
        self.avg_inference_time = 0.0

        # Thread lock for state
        self.lock = threading.Lock()

        logger.info(
            f"InferenceEngine initialized: fps={capture_fps}, "
            f"interval={inference_interval}s, seq_len={sequence_length}"
        )

    def start(self):
        """Start capture and inference threads"""
        if self.is_running:
            logger.warning("InferenceEngine already running")
            return

        self.is_running = True

        # Start capture
        self.capture.start(target_fps=self.capture_fps)

        # Start threads
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)

        self.capture_thread.start()
        self.inference_thread.start()

        logger.info("InferenceEngine started")

    def stop(self):
        """Stop capture and inference"""
        if not self.is_running:
            return

        self.is_running = False

        # Wait for threads to finish
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)

        # Stop capture
        self.capture.stop()

        logger.info("InferenceEngine stopped")

    def _capture_loop(self):
        """Capture loop: grabs frames and encodes to latents"""
        logger.info("Capture loop started")

        while self.is_running:
            try:
                # Grab frame
                frame = self.capture.grab_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue

                # Encode frame to latent
                with torch.no_grad():
                    # Convert to tensor: [H, W, C] -> [1, C, H, W]
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    frame_tensor = frame_tensor.unsqueeze(0).to(self.device)

                    # Encode
                    latent = self.model.encoder(frame_tensor).squeeze(0)  # [D]

                # Store in buffer
                with self.lock:
                    self.buffer.push(frame, latent)
                    self.total_frames += 1

            except Exception as e:
                logger.error(f"Error in capture loop: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("Capture loop stopped")

    def _inference_loop(self):
        """Inference loop: periodically predicts EV"""
        logger.info("Inference loop started")

        while self.is_running:
            try:
                time.sleep(self.inference_interval)

                # Check if buffer has enough frames
                with self.lock:
                    buffer_size = len(self.buffer)
                    is_full = self.buffer.is_full

                if buffer_size < self.sequence_length:
                    logger.debug(
                        f"Buffer not ready: {buffer_size}/{self.sequence_length}"
                    )
                    continue

                # Get latent sequence
                with self.lock:
                    z_seq = self.buffer.get_last_n_latents(
                        self.sequence_length, self.device
                    )

                if z_seq is None:
                    continue

                # Add batch dimension: [L, D] -> [1, L, D]
                z_seq = z_seq.unsqueeze(0)

                # Predict EV
                start_time = time.time()

                with torch.no_grad():
                    quantile = config.inference.quantile_idx
                    ev_stay, ev_extract, delta_ev = self.model.predict_ev(
                        latents=z_seq, quantile=quantile
                    )

                inference_time = (time.time() - start_time) * 1000  # ms

                # Extract scalar values
                ev_stay = ev_stay.item()
                ev_extract = ev_extract.item()
                delta_ev = delta_ev.item()

                # Determine recommendation
                threshold = 20.0  # TODO: Make configurable
                if delta_ev > threshold:
                    recommendation = "stay"
                    confidence = min(delta_ev / 100.0, 1.0)
                elif delta_ev < -threshold:
                    recommendation = "extract"
                    confidence = min(abs(delta_ev) / 100.0, 1.0)
                else:
                    recommendation = "neutral"
                    confidence = 0.5

                # Create prediction object
                prediction = EVPrediction(
                    ev_stay=ev_stay,
                    ev_extract=ev_extract,
                    delta_ev=delta_ev,
                    recommendation=recommendation,
                    confidence=confidence,
                    timestamp=time.time(),
                    quantile=quantile,
                    inference_time_ms=inference_time,
                    buffer_size=buffer_size,
                    is_buffer_full=is_full,
                )

                # Update state
                with self.lock:
                    self.last_prediction = prediction
                    self.total_inferences += 1
                    self.avg_inference_time = (
                        self.avg_inference_time * 0.9 + inference_time * 0.1
                    )

                logger.debug(
                    f"EV prediction: stay={ev_stay:.1f}, extract={ev_extract:.1f}, "
                    f"delta={delta_ev:.1f}, rec={recommendation}, time={inference_time:.1f}ms"
                )

            except Exception as e:
                logger.error(f"Error in inference loop: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("Inference loop stopped")

    def get_latest_prediction(self) -> Optional[EVPrediction]:
        """Get latest EV prediction (thread-safe)"""
        with self.lock:
            return self.last_prediction

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        with self.lock:
            return {
                "is_running": self.is_running,
                "total_frames": self.total_frames,
                "total_inferences": self.total_inferences,
                "buffer_size": len(self.buffer),
                "buffer_capacity": self.buffer.frame_buffer.capacity,
                "buffer_full": self.buffer.is_full,
                "capture_fps": self.capture.get_fps(),
                "avg_inference_time_ms": self.avg_inference_time,
            }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def test_inference_engine():
    """Test inference engine with fallback capture"""
    print("Testing InferenceEngine...")

    from arcx.ml.model import EVModel

    # Create model (no pretrained for speed)
    model = EVModel(encoder_pretrained=False)
    model.to(device_manager.device)

    # Create engine
    engine = InferenceEngine(
        model=model,
        capture_fps=8,
        inference_interval=0.5,
        sequence_length=32,
    )

    # Run for a few seconds
    with engine:
        print("Engine started, running for 5 seconds...")
        time.sleep(5)

        # Get prediction
        prediction = engine.get_latest_prediction()
        if prediction:
            print(f"Latest prediction: {prediction}")

        # Get stats
        stats = engine.get_stats()
        print(f"Stats: {stats}")

    print("✓ InferenceEngine test passed")


if __name__ == "__main__":
    test_inference_engine()
