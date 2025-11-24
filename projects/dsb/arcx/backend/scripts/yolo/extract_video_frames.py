"""Extract frames from video for YOLO annotation.

This tool extracts frames from gameplay videos to create a dataset
for YOLO training. It's optimized for extraction screen capture videos.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: Optional[float] = None,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    max_frames: Optional[int] = None,
    prefix: str = "frame",
    quality: int = 95,
    resize: Optional[tuple] = None,
):
    """
    Extract frames from video.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save extracted frames
        fps: Target FPS for extraction (None = original FPS)
        start_sec: Start time in seconds
        end_sec: End time in seconds (None = end of video)
        max_frames: Maximum number of frames to extract
        prefix: Prefix for output filenames
        quality: JPEG quality (0-100)
        resize: Resize frames to (width, height) tuple
    """
    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    logger.info(f"Video: {video_path.name}")
    logger.info(f"  FPS: {video_fps:.2f}")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  Duration: {duration:.2f}s")

    # Calculate frame extraction parameters
    if fps is None:
        fps = video_fps

    frame_interval = max(1, int(video_fps / fps))
    start_frame = int(start_sec * video_fps)
    end_frame = int(end_sec * video_fps) if end_sec else total_frames

    logger.info(f"Extraction settings:")
    logger.info(f"  Target FPS: {fps:.2f}")
    logger.info(f"  Frame interval: {frame_interval}")
    logger.info(f"  Start frame: {start_frame}")
    logger.info(f"  End frame: {end_frame}")
    if max_frames:
        logger.info(f"  Max frames: {max_frames}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    extracted = 0
    frame_num = start_frame
    saved_count = 0

    pbar = tqdm(total=min(end_frame - start_frame, max_frames or float('inf')),
                desc="Extracting frames")

    while frame_num < end_frame:
        ret, frame = cap.read()

        if not ret:
            break

        # Extract every Nth frame
        if (frame_num - start_frame) % frame_interval == 0:
            # Resize if needed
            if resize:
                frame = cv2.resize(frame, resize)

            # Save frame
            filename = f"{prefix}_{saved_count:06d}.png"
            filepath = output_dir / filename

            cv2.imwrite(
                str(filepath),
                frame,
                [cv2.IMWRITE_PNG_COMPRESSION, 9 - (quality // 11)]  # Convert quality
            )

            saved_count += 1
            pbar.update(1)

            # Check max frames limit
            if max_frames and saved_count >= max_frames:
                break

        frame_num += 1

    pbar.close()
    cap.release()

    logger.info(f"✓ Extracted {saved_count} frames to {output_dir}")


def extract_frames_from_multiple_videos(
    video_dir: Path,
    output_dir: Path,
    **kwargs
):
    """
    Extract frames from all videos in a directory.

    Args:
        video_dir: Directory containing video files
        output_dir: Base output directory
        **kwargs: Additional arguments for extract_frames()
    """
    # Find video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = []

    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"*{ext}"))
        video_files.extend(video_dir.glob(f"*{ext.upper()}"))

    if not video_files:
        logger.error(f"No video files found in {video_dir}")
        return

    logger.info(f"Found {len(video_files)} video files")

    # Extract frames from each video
    for video_path in sorted(video_files):
        # Create subdirectory for each video
        video_output_dir = output_dir / video_path.stem

        logger.info(f"\nProcessing: {video_path.name}")

        extract_frames(
            video_path=video_path,
            output_dir=video_output_dir,
            prefix=video_path.stem,
            **kwargs
        )


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from video for YOLO annotation"
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Path to input video file",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        help="Directory containing multiple videos",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for extracted frames",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="Target FPS for extraction (default: original FPS)",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Start time in seconds (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=float,
        help="End time in seconds (default: end of video)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum number of frames to extract",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame",
        help="Prefix for output filenames (default: frame)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality 0-100 (default: 95)",
    )
    parser.add_argument(
        "--resize",
        type=str,
        help="Resize frames to WIDTHxHEIGHT (e.g., 1920x1080)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        help="Extract one frame every N seconds (alternative to --fps)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.video and not args.video_dir:
        parser.error("Either --video or --video-dir must be specified")

    if args.video and args.video_dir:
        parser.error("Cannot specify both --video and --video-dir")

    # Parse resize
    resize = None
    if args.resize:
        try:
            width, height = map(int, args.resize.lower().split('x'))
            resize = (width, height)
        except ValueError:
            parser.error("Invalid resize format. Use WIDTHxHEIGHT (e.g., 1920x1080)")

    # Calculate FPS from interval if specified
    fps = args.fps
    if args.interval and not fps:
        fps = 1.0 / args.interval

    # Extract frames
    if args.video:
        extract_frames(
            video_path=args.video,
            output_dir=args.output,
            fps=fps,
            start_sec=args.start,
            end_sec=args.end,
            max_frames=args.max_frames,
            prefix=args.prefix,
            quality=args.quality,
            resize=resize,
        )
    else:
        extract_frames_from_multiple_videos(
            video_dir=args.video_dir,
            output_dir=args.output,
            fps=fps,
            start_sec=args.start,
            end_sec=args.end,
            max_frames=args.max_frames,
            quality=args.quality,
            resize=resize,
        )


if __name__ == "__main__":
    main()
