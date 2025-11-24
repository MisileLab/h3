"""Prepare YOLO dataset from raw screenshots and annotations."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

from arcx.valuation.detector import ItemDetector

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_annotations_to_yolo(
    annotations_file: Path,
    images_dir: Path,
    output_dir: Path,
    split: str = "train",
):
    """
    Convert custom annotations to YOLO format.

    Expected annotation format (JSON):
    {
        "image_filename.png": {
            "items": [
                {
                    "item_type": "weapon",
                    "rarity": "epic",
                    "bbox": [x_min, y_min, x_max, y_max]
                },
                ...
            ],
            "image_width": 1920,
            "image_height": 1080
        },
        ...
    }

    YOLO format (one .txt per image):
    <class_id> <x_center> <y_center> <width> <height>
    (all normalized to 0-1)

    Args:
        annotations_file: Path to JSON annotations file
        images_dir: Directory containing images
        output_dir: Output directory for YOLO dataset
        split: Dataset split (train/val/test)
    """
    # Load annotations
    with open(annotations_file) as f:
        annotations = json.load(f)

    # Create output directories
    images_output = output_dir / "images" / split
    labels_output = output_dir / "labels" / split
    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)

    detector = ItemDetector()  # For class mapping

    processed = 0
    skipped = 0

    for image_filename, data in annotations.items():
        image_path = images_dir / image_filename

        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            skipped += 1
            continue

        img_width = data["image_width"]
        img_height = data["image_height"]

        # Convert annotations
        yolo_lines = []
        for item in data["items"]:
            item_type = item["item_type"]
            rarity = item["rarity"]
            bbox = item["bbox"]  # [x_min, y_min, x_max, y_max]

            # Get class ID
            try:
                class_id = detector.get_class_id(item_type, rarity)
            except ValueError as e:
                logger.warning(f"Invalid item in {image_filename}: {e}")
                continue

            # Convert to YOLO format (normalized center x, y, width, height)
            x_min, y_min, x_max, y_max = bbox
            x_center = (x_min + x_max) / 2.0 / img_width
            y_center = (y_min + y_max) / 2.0 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )

        if not yolo_lines:
            logger.warning(f"No valid annotations for {image_filename}")
            skipped += 1
            continue

        # Copy image
        shutil.copy(image_path, images_output / image_filename)

        # Save label file
        label_filename = image_path.stem + ".txt"
        label_path = labels_output / label_filename
        label_path.write_text("\n".join(yolo_lines))

        processed += 1

    logger.info(
        f"Processed {processed} images for {split} split ({skipped} skipped)"
    )


def split_dataset(
    annotations_file: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> Dict[str, List[str]]:
    """
    Split annotations into train/val/test sets.

    Args:
        annotations_file: Path to annotations JSON
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio

    Returns:
        Dictionary with keys 'train', 'val', 'test' mapping to image filenames
    """
    import random

    with open(annotations_file) as f:
        annotations = json.load(f)

    image_filenames = list(annotations.keys())
    random.shuffle(image_filenames)

    n_total = len(image_filenames)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": image_filenames[:n_train],
        "val": image_filenames[n_train : n_train + n_val],
        "test": image_filenames[n_train + n_val :],
    }

    logger.info(f"Dataset split: train={len(splits['train'])}, "
                f"val={len(splits['val'])}, test={len(splits['test'])}")

    return splits


def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset")
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Path to annotations JSON file",
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing images",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/yolo_items"),
        help="Output directory for YOLO dataset",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation set ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )

    args = parser.parse_args()

    import random
    random.seed(args.seed)

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        logger.error(f"Ratios must sum to 1.0 (got {total_ratio})")
        return

    # Load and split annotations
    logger.info("Loading annotations...")
    with open(args.annotations) as f:
        annotations = json.load(f)

    splits = split_dataset(
        args.annotations,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
    )

    # Convert each split
    for split_name in ["train", "val", "test"]:
        logger.info(f"Processing {split_name} split...")

        # Create split-specific annotations
        split_annotations = {
            filename: annotations[filename]
            for filename in splits[split_name]
        }

        # Save temporary split annotations
        temp_file = args.output / f"{split_name}_annotations.json"
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_file, "w") as f:
            json.dump(split_annotations, f, indent=2)

        # Convert to YOLO format
        convert_annotations_to_yolo(
            temp_file,
            args.images,
            args.output,
            split_name,
        )

        # Clean up temp file
        temp_file.unlink()

    logger.info(f"Dataset prepared at: {args.output}")
    logger.info("Next step: Run train_item_detector.py")


if __name__ == "__main__":
    main()
