"""Training worker for Vast.ai instances.

This module provides the worker process that runs on Vast.ai instances:
- Loads checkpoints from mounted storage
- Runs self-play training
- Saves checkpoints periodically
- Handles graceful shutdown on signals
"""

import signal
import sys
from pathlib import Path
from typing import Any

from tsgb.checkpoint import find_latest_checkpoint
from tsgb.logging import configure_logging, get_logger
from tsgb.models import HuggingFaceLM, LLMRole, get_accelerator
from tsgb.settings import Settings, get_settings
from tsgb.trainer import SelfPlayTrainer, TrainConfig

logger = get_logger(__name__)


class GracefulShutdown:
    """Handler for graceful shutdown on SIGINT/SIGTERM."""

    def __init__(self) -> None:
        self.should_stop = False
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)

    def __enter__(self) -> "GracefulShutdown":
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)
        return self

    def __exit__(self, *args: object) -> None:
        signal.signal(signal.SIGINT, self._original_sigint)
        signal.signal(signal.SIGTERM, self._original_sigterm)

    def _handler(self, signum: int, frame: Any) -> None:
        """Signal handler that sets the stop flag."""
        sig_name = signal.Signals(signum).name
        logger.info("shutdown_signal_received", signal=sig_name)
        self.should_stop = True


def find_checkpoint_path(
    resume_path: str | Path | None,
    local_fallback: str | Path | None,
) -> Path | None:
    """Find the best checkpoint path to resume from.

    Args:
        resume_path: Primary path to look for checkpoints.
        local_fallback: Fallback path if primary is unavailable.

    Returns:
        Path to checkpoint directory, or None if not found.
    """
    paths_to_try = []

    if resume_path:
        paths_to_try.append(Path(resume_path))

    if local_fallback:
        paths_to_try.append(Path(local_fallback))

    for path in paths_to_try:
        if path.exists():
            checkpoint = find_latest_checkpoint(path)
            if checkpoint:
                logger.info("checkpoint_found", path=str(checkpoint))
                return checkpoint
            else:
                logger.debug("no_checkpoint_in_path", path=str(path))
        else:
            logger.debug("path_not_exists", path=str(path))

    return None


def init_trackio_run(
    enabled: bool,
    project: str,
    space_id: str | None,
    initial_config: dict[str, Any],
) -> Any | None:
    """Initialize Trackio run if enabled and available."""
    if not enabled:
        return None

    try:
        import trackio  # type: ignore
    except ImportError:
        logger.warning("trackio_not_installed")
        return None

    try:
        kwargs: dict[str, Any] = {}
        if space_id:
            kwargs["trackio_space_id"] = space_id

        run = trackio.init(project=project, **kwargs)
        if hasattr(trackio, "config"):
            try:
                trackio.config.update(initial_config, allow_val_change=True)
            except Exception as config_error:  # pragma: no cover - best effort
                logger.debug("trackio_config_update_failed", error=str(config_error))

        logger.info("trackio_initialized", project=project, space_id=space_id)
        return run
    except Exception as e:  # pragma: no cover - runtime safeguard
        logger.error("trackio_init_failed", error=str(e))
        return None


def run_worker(
    resume_path: str | Path | None = None,
    local_fallback: str | Path | None = None,
    model_name: str | None = None,
    total_episodes: int = 1000,
    checkpoint_interval: int = 100,
    log_interval: int = 10,
    enable_trackio: bool | None = None,
    trackio_project: str | None = None,
    trackio_space_id: str | None = None,
    settings: Settings | None = None,
) -> None:
    """Run the training worker.

    Args:
        resume_path: Path to look for checkpoints (e.g., mounted WebDAV).
        local_fallback: Fallback checkpoint path if resume_path unavailable.
        model_name: Model name to use (default: from settings).
        total_episodes: Total training episodes.
        checkpoint_interval: Save checkpoint every N episodes.
        log_interval: Log metrics every N episodes.
        settings: Application settings.
    """
    settings = settings or get_settings()

    trackio_enabled = settings.trackio_enabled if enable_trackio is None else enable_trackio
    trackio_project_name = trackio_project or settings.trackio_project
    trackio_space = trackio_space_id or settings.trackio_space_id

    # Configure logging
    configure_logging(mode=settings.log_mode)  # type: ignore

    model = model_name or settings.default_model_name

    logger.info(
        "worker_starting",
        resume_path=str(resume_path) if resume_path else None,
        local_fallback=str(local_fallback) if local_fallback else None,
        model_name=model,
        trackio_enabled=trackio_enabled,
        trackio_project=trackio_project_name,
    )

    trackio_run: Any | None = None

    # Determine checkpoint directory
    if resume_path:
        checkpoint_dir = Path(resume_path)
    elif local_fallback:
        checkpoint_dir = Path(local_fallback)
    else:
        checkpoint_dir = Path(settings.local_checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Find existing checkpoint to resume from
    resume_from = find_checkpoint_path(resume_path, local_fallback)

    # Create training config
    config = TrainConfig(
        total_episodes=total_episodes,
        checkpoint_interval=checkpoint_interval,
        log_interval=log_interval,
        checkpoint_dir=str(checkpoint_dir),
        seed=42,
    )

    trackio_run = init_trackio_run(
        enabled=trackio_enabled,
        project=trackio_project_name,
        space_id=trackio_space,
        initial_config={
            "model/name": model,
            "training/total_episodes": config.total_episodes,
            "training/checkpoint_interval": config.checkpoint_interval,
            "training/log_interval": config.log_interval,
            "checkpoint_dir": str(checkpoint_dir),
            "resume_from": str(resume_from) if resume_from else "scratch",
        },
    )

    # Initialize accelerator before loading models so all models share the same config
    accelerator = get_accelerator(
        mixed_precision="fp16" if config.use_mixed_precision else None,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
    )

    # Load models
    logger.info("loading_models", model_name=model)
    attacker = HuggingFaceLM.from_pretrained(
        model,
        role=LLMRole.ATTACKER,
        accelerator=accelerator,
        use_mixed_precision=config.use_mixed_precision,
        gradient_checkpointing=config.enable_gradient_checkpointing,
    )
    guard = HuggingFaceLM.from_pretrained(
        model,
        role=LLMRole.GUARD,
        accelerator=accelerator,
        use_mixed_precision=config.use_mixed_precision,
        gradient_checkpointing=config.enable_gradient_checkpointing,
    )
    target = HuggingFaceLM.from_pretrained(
        model,
        role=LLMRole.TARGET,
        accelerator=accelerator,
        use_mixed_precision=config.use_mixed_precision,
        gradient_checkpointing=config.enable_gradient_checkpointing,
    )

    trainer = SelfPlayTrainer(
        attacker_model=attacker,
        guard_model=guard,
        target_model=target,
        settings=settings,
        config=config,
        trackio_run=trackio_run,
    )

    # Set up graceful shutdown
    with GracefulShutdown() as shutdown:
        try:
            # Run training
            logger.info(
                "training_starting",
                resume_from=str(resume_from) if resume_from else "scratch",
                total_episodes=config.total_episodes,
            )

            trainer.train(resume_from=resume_from)

            if trackio_run is not None:
                try:
                    trackio_run.log(
                        {
                            "training/status": "completed",
                            "training/episode": trainer.episode_index,
                        },
                        step=trainer.global_step,
                    )
                except Exception as e:  # pragma: no cover - best effort logging
                    logger.debug("trackio_completion_log_failed", error=str(e))

        except KeyboardInterrupt:
            logger.info("training_interrupted_by_user")
            if trackio_run is not None:
                try:
                    trackio_run.log({"training/status": "interrupted"}, step=trainer.global_step)
                except Exception as e:  # pragma: no cover - best effort logging
                    logger.debug("trackio_interrupt_log_failed", error=str(e))

        except Exception as e:
            logger.error("training_error", error=str(e), exc_info=True)
            if trackio_run is not None:
                try:
                    trackio_run.log({"training/status": "errored", "error": str(e)})
                except Exception as log_error:  # pragma: no cover - best effort logging
                    logger.debug("trackio_error_log_failed", error=str(log_error))
            raise

        finally:
            # Save final checkpoint
            logger.info("saving_final_checkpoint")
            try:
                trainer._save_checkpoint()
            except Exception as e:
                logger.error("final_checkpoint_failed", error=str(e))

            if trackio_run is not None:
                try:
                    trackio_run.finish()
                    logger.info("trackio_run_finished")
                except Exception as e:
                    logger.warning("trackio_finish_failed", error=str(e))

    logger.info("worker_finished")


def main() -> None:
    """Main entry point for worker process."""
    import argparse

    parser = argparse.ArgumentParser(description="TSGB Training Worker")
    parser.add_argument(
        "--resume-path",
        type=str,
        help="Path to checkpoint directory (e.g., mounted WebDAV)",
    )
    parser.add_argument(
        "--local-fallback",
        type=str,
        help="Fallback checkpoint path if resume-path unavailable",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name to use (default: from settings)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Total training episodes",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Save checkpoint every N episodes",
    )

    args = parser.parse_args()

    run_worker(
        resume_path=args.resume_path,
        local_fallback=args.local_fallback,
        model_name=args.model,
        total_episodes=args.episodes,
        checkpoint_interval=args.checkpoint_interval,
    )


if __name__ == "__main__":
    main()
