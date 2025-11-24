"""Filter extracted frames to find extraction screens.

This tool helps filter frames to find only extraction/loot screens,
reducing the annotation workload.
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_similar_to_template(
    frame: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.7,
) -> bool:
    """
    Check if frame is similar to template using template matching.

    Args:
        frame: Input frame
        template: Template image (extraction screen sample)
        threshold: Similarity threshold (0-1)

    Returns:
        True if similar enough
    """
    # Convert to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Resize template if needed
    if frame_gray.shape != template_gray.shape:
        template_gray = cv2.resize(template_gray, (frame_gray.shape[1], frame_gray.shape[0]))

    # Template matching
    result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)

    return max_val >= threshold


def has_ui_elements(
    frame: np.ndarray,
    color_ranges: list = None,
    min_pixels: int = 1000,
) -> bool:
    """
    Check if frame contains UI elements by color detection.

    Args:
        frame: Input frame
        color_ranges: List of (lower, upper) BGR color ranges to detect
        min_pixels: Minimum number of pixels to consider as detected

    Returns:
        True if UI elements detected
    """
    if color_ranges is None:
        # Default: detect bright UI elements (white/yellow/gold)
        color_ranges = [
            (np.array([200, 200, 200]), np.array([255, 255, 255])),  # White
            (np.array([0, 200, 200]), np.array([50, 255, 255])),    # Yellow/Gold
        ]

    total_pixels = 0

    for lower, upper in color_ranges:
        mask = cv2.inRange(frame, lower, upper)
        total_pixels += cv2.countNonZero(mask)

    return total_pixels >= min_pixels


def filter_frames_interactive(
    input_dir: Path,
    output_dir: Path,
    preview: bool = True,
):
    """
    Interactively filter frames (manual selection).

    Args:
        input_dir: Directory containing frames
        output_dir: Directory to save selected frames
        preview: Show preview window
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_files = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")))

    if not image_files:
        logger.error(f"No images found in {input_dir}")
        return

    logger.info(f"Found {len(image_files)} images")
    logger.info("Controls:")
    logger.info("  SPACE - Select and save")
    logger.info("  S - Skip")
    logger.info("  Q - Quit")
    logger.info("  A - Auto-select next 10")

    if preview:
        cv2.namedWindow("Frame Filter", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Frame Filter", 1280, 720)

    selected = 0
    auto_mode = 0

    for idx, img_path in enumerate(image_files):
        # Load image
        frame = cv2.imread(str(img_path))

        if frame is None:
            continue

        # Show preview
        if preview:
            # Add text overlay
            display = frame.copy()
            text = f"[{idx+1}/{len(image_files)}] {img_path.name}"
            cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2)
            cv2.imshow("Frame Filter", display)

            # Wait for key
            if auto_mode > 0:
                key = ord(' ')  # Auto-select
                auto_mode -= 1
                cv2.waitKey(100)
            else:
                key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                # Select
                output_path = output_dir / img_path.name
                shutil.copy(img_path, output_path)
                selected += 1
                logger.info(f"Selected: {img_path.name} ({selected} total)")
            elif key == ord('a'):
                # Auto-select mode
                auto_mode = 10
                logger.info("Auto-selecting next 10 frames")
                # Select current frame
                output_path = output_dir / img_path.name
                shutil.copy(img_path, output_path)
                selected += 1
            # 's' or any other key = skip
        else:
            # Non-interactive mode: copy all
            output_path = output_dir / img_path.name
            shutil.copy(img_path, output_path)
            selected += 1

    if preview:
        cv2.destroyAllWindows()

    logger.info(f"✓ Selected {selected} frames to {output_dir}")


def filter_frames_automatic(
    input_dir: Path,
    output_dir: Path,
    template_path: Optional[Path] = None,
    similarity_threshold: float = 0.7,
):
    """
    Automatically filter frames using template matching or heuristics.

    Args:
        input_dir: Directory containing frames
        output_dir: Directory to save filtered frames
        template_path: Path to template image (extraction screen sample)
        similarity_threshold: Similarity threshold for template matching
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_files = sorted(list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")))

    if not image_files:
        logger.error(f"No images found in {input_dir}")
        return

    # Load template if provided
    template = None
    if template_path and template_path.exists():
        template = cv2.imread(str(template_path))
        logger.info(f"Loaded template: {template_path}")

    logger.info(f"Processing {len(image_files)} images...")

    selected = 0

    for img_path in tqdm(image_files, desc="Filtering"):
        frame = cv2.imread(str(img_path))

        if frame is None:
            continue

        # Filter logic
        is_extraction = False

        if template is not None:
            # Template matching
            is_extraction = is_similar_to_template(frame, template, similarity_threshold)
        else:
            # Heuristic: check for UI elements
            is_extraction = has_ui_elements(frame)

        if is_extraction:
            output_path = output_dir / img_path.name
            shutil.copy(img_path, output_path)
            selected += 1

    logger.info(f"✓ Filtered {selected}/{len(image_files)} frames to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter frames to find extraction screens"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing frames",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for filtered frames",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "auto"],
        default="interactive",
        help="Filtering mode (default: interactive)",
    )
    parser.add_argument(
        "--template",
        type=Path,
        help="Template image for automatic filtering",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for template matching (default: 0.7)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable preview window (copy all frames)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input directory not found: {args.input}")
        return

    if args.mode == "interactive":
        filter_frames_interactive(
            input_dir=args.input,
            output_dir=args.output,
            preview=not args.no_preview,
        )
    else:
        filter_frames_automatic(
            input_dir=args.input,
            output_dir=args.output,
            template_path=args.template,
            similarity_threshold=args.threshold,
        )


if __name__ == "__main__":
    main()
