"""Convert YOLO .pt model to safetensors format."""

import argparse
import logging
from pathlib import Path

import torch
from safetensors.torch import save_file, load_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_pt_to_safetensors(pt_path: Path, output_path: Path):
    """
    Convert YOLO .pt model to safetensors format.

    Args:
        pt_path: Path to .pt model file
        output_path: Path to save .safetensors file
    """
    logger.info(f"Loading model from {pt_path}...")

    # Load the .pt checkpoint
    checkpoint = torch.load(pt_path, map_location="cpu")

    # Extract model state dict
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"].state_dict()
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            # Assume the checkpoint is already a state dict
            state_dict = checkpoint
    else:
        # If it's a model object directly
        state_dict = checkpoint.state_dict()

    # Convert all tensors to contiguous format (required for safetensors)
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}

    logger.info(f"Saving to {output_path}...")
    save_file(state_dict, output_path)

    # Verify the conversion
    logger.info("Verifying conversion...")
    loaded_state_dict = load_file(output_path)

    if len(state_dict) != len(loaded_state_dict):
        logger.error("Mismatch in number of parameters!")
        return False

    logger.info(f"✓ Successfully converted {len(state_dict)} parameters")
    logger.info(f"  Original size: {pt_path.stat().st_size / 1024 / 1024:.2f} MB")
    logger.info(f"  Safetensors size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO .pt model to safetensors"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input .pt model file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to output .safetensors file (auto-generated if not provided)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        return

    # Auto-generate output path if not provided
    if args.output is None:
        args.output = args.input.with_suffix(".safetensors")

    success = convert_pt_to_safetensors(args.input, args.output)

    if success:
        logger.info("✓ Conversion complete!")
    else:
        logger.error("✗ Conversion failed!")
        exit(1)


if __name__ == "__main__":
    main()
