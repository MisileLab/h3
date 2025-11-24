"""Train YOLO11 model for item detection."""

import argparse
import logging
from pathlib import Path

import torch
from safetensors.torch import save_file
from ultralytics import YOLO

from arcx.valuation.config import ITEM_TYPES, RARITY_LEVELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_yaml_config(data_dir: Path, output_path: Path):
    """
    Create YOLO dataset configuration file.

    Args:
        data_dir: Path to dataset directory
        output_path: Path to save config YAML
    """
    num_classes = len(ITEM_TYPES) * len(RARITY_LEVELS)

    # Build class names list
    class_names = []
    for item_type in ITEM_TYPES:
        for rarity in RARITY_LEVELS:
            class_names.append(f"{item_type}_{rarity}")

    yaml_content = f"""# ArcX Item Detection Dataset
path: {data_dir.absolute()}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: {num_classes}

# Class names
names:
"""
    for idx, name in enumerate(class_names):
        yaml_content += f"  {idx}: {name}\n"

    output_path.write_text(yaml_content)
    logger.info(f"Created dataset config at {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 item detector")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/yolo_items"),
        help="Path to YOLO dataset directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        choices=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
        help="Base YOLO11 model to use",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use (cuda device id or 'cpu')",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/yolo_train"),
        help="Project directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="item_detector",
        help="Experiment name",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )

    args = parser.parse_args()

    # Validate dataset directory
    if not args.data_dir.exists():
        logger.error(f"Dataset directory not found: {args.data_dir}")
        logger.info("Please prepare dataset using prepare_dataset.py first")
        return

    # Create dataset config
    config_path = args.data_dir / "dataset.yaml"
    create_yaml_config(args.data_dir, config_path)

    # Load YOLO model
    logger.info(f"Loading YOLO11 model: {args.model}")
    model = YOLO(args.model)

    # Train
    logger.info("Starting training...")
    results = model.train(
        data=str(config_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        resume=args.resume,
        plots=True,
        save=True,
        val=True,
        # Optimization settings
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
    )

    logger.info("Training complete!")
    logger.info(f"Results saved to: {args.project / args.name}")

    # Export best model
    best_model_path = args.project / args.name / "weights" / "best.pt"
    if best_model_path.exists():
        logger.info(f"Best model: {best_model_path}")

        # Convert to safetensors
        logger.info("Converting to safetensors format...")
        safetensors_path = best_model_path.with_suffix(".safetensors")

        try:
            # Load checkpoint
            checkpoint = torch.load(best_model_path, map_location="cpu")

            # Extract model state dict
            if isinstance(checkpoint, dict):
                if "model" in checkpoint:
                    state_dict = checkpoint["model"].state_dict()
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint.state_dict()

            # Convert to contiguous format
            state_dict = {k: v.contiguous() for k, v in state_dict.items()}

            # Save as safetensors
            save_file(state_dict, safetensors_path)

            logger.info(f"✓ Safetensors model saved: {safetensors_path}")
            logger.info(f"  .pt size: {best_model_path.stat().st_size / 1024 / 1024:.2f} MB")
            logger.info(f"  .safetensors size: {safetensors_path.stat().st_size / 1024 / 1024:.2f} MB")
            logger.info("")
            logger.info("Copy to production:")
            logger.info(f"  cp {safetensors_path} models/item_detector_yolo11.safetensors")

        except Exception as e:
            logger.error(f"Failed to convert to safetensors: {e}")
            logger.info("You can manually convert later using convert_to_safetensors.py")


if __name__ == "__main__":
    main()
