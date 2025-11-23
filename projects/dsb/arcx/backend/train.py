"""Training script for EV model

Usage:
    python train.py --epochs 50 --batch-size 32 --lr 1e-4
"""

import argparse
import logging
import sys
from pathlib import Path

from arcx.ml.model import EVModel
from arcx.data.dataset import create_dataloaders
from arcx.training.trainer import Trainer
from arcx.device import device_manager
from arcx.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train EV model")

    # Data args
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=config.data.log_dir,
        help="Data directory containing Parquet files",
    )

    # Model args
    parser.add_argument(
        "--encoder-backbone",
        type=str,
        default=config.model.encoder_backbone,
        choices=["resnet34", "resnet50"],
        help="Encoder backbone architecture",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=config.model.encoder_dim,
        help="Latent vector dimension",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=config.model.hidden_dim,
        help="Hidden dimension for Q-net",
    )
    parser.add_argument(
        "--num-quantiles",
        type=int,
        default=config.model.num_quantiles,
        help="Number of quantiles",
    )
    parser.add_argument(
        "--temporal-encoder",
        type=str,
        default=config.model.temporal_encoder,
        choices=["gru", "transformer"],
        help="Temporal encoder type",
    )

    # Training args
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.model.batch_size,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=config.model.learning_rate,
        help="Learning rate",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data for training",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )

    # Other args
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint directory",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use ImageNet pretrained encoder",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Don't use pretrained encoder",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("ArcX EV Model Training")
    logger.info("=" * 80)
    logger.info(f"Device: {device_manager.device} ({device_manager.backend.value})")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info("=" * 80)

    # Check data directory
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        logger.error("Please run the system to collect data first, or specify correct --data-dir")
        sys.exit(1)

    parquet_files = list(args.data_dir.glob("decisions_*.parquet"))
    if not parquet_files:
        logger.error(f"No decision Parquet files found in {args.data_dir}")
        sys.exit(1)

    logger.info(f"Found {len(parquet_files)} data files")

    # Create dataloaders
    logger.info("Creating dataloaders...")
    try:
        train_loader, val_loader = create_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            train_split=args.train_split,
            num_workers=args.num_workers,
        )
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}")
        sys.exit(1)

    # Create model
    logger.info("Creating model...")
    model = EVModel(
        encoder_backbone=args.encoder_backbone,
        latent_dim=args.latent_dim,
        encoder_pretrained=args.pretrained,
        hidden_dim=args.hidden_dim,
        num_quantiles=args.num_quantiles,
        temporal_encoder=args.temporal_encoder,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
    )

    # Resume if requested
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        try:
            trainer.load_checkpoint(args.resume)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            sys.exit(1)

    # Train
    logger.info("Starting training...")
    try:
        trainer.train(num_epochs=args.epochs, save_every=args.save_every)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint()
        logger.info("Checkpoint saved")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Training complete!")
    logger.info(f"Best model saved to: {config.model.encoder_path.parent}")


if __name__ == "__main__":
    main()
