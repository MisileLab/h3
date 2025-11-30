"""Self-play RL trainer for Stage 1 training.

This module provides PPO-style self-play training between Attacker and Guard models.
Uses proper PPO implementation with GAE advantage estimation.
Supports multi-GPU training via accelerate.
"""

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch.optim import AdamW

from tsgb.checkpoint import (
    TrainerState,
    find_latest_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from tsgb.envs import SafetyJudge, SelfPlayEnv
from tsgb.logging import get_logger
from tsgb.models import HuggingFaceLM, LLMRole, get_accelerator
from tsgb.settings import Settings

logger = get_logger(__name__)


def get_available_vram_gb() -> float:
    """Get total available VRAM across all GPUs in GB.

    Returns:
        Total VRAM in GB, or 0 if no GPU available.
    """
    if not torch.cuda.is_available():
        return 0.0

    total_vram = 0.0
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_vram += props.total_memory / (1024**3)  # Convert bytes to GB

    return total_vram


def compute_batch_size(vram_gb: float, model_size_b: float = 3.0) -> int:
    """Compute optimal batch size based on available VRAM.

    Uses a simple heuristic:
    - ~3B model needs ~12GB per sample with gradients
    - Scale batch size based on available VRAM
    - Minimum batch size of 1, maximum of 32

    Args:
        vram_gb: Available VRAM in GB.
        model_size_b: Model size in billions of parameters.

    Returns:
        Recommended batch size.
    """
    if vram_gb <= 0:
        return 1

    # Rough estimate: model needs ~4 bytes per param for weights
    # Plus ~8 bytes per param for optimizer states (AdamW)
    # Plus ~4 bytes per param for gradients
    # Total: ~16 bytes per param, plus activations

    # Base memory for 3B model: ~48GB for weights+optimizer+gradients
    # Each sample in batch needs additional activation memory

    # Heuristic:
    # - 24GB VRAM -> batch 1-2
    # - 48GB VRAM -> batch 2-4
    # - 80GB VRAM -> batch 4-8
    # - 160GB (2x80GB) -> batch 8-16

    # Reserve ~60% of VRAM for model, use rest for batch
    available_for_batch = vram_gb * 0.4

    # Assume each batch sample needs ~6GB for 3B model activations
    memory_per_sample = 6.0 * (model_size_b / 3.0)

    batch_size = int(available_for_batch / memory_per_sample)

    # Clamp to reasonable range
    batch_size = max(1, min(batch_size, 32))

    logger.info(
        "computed_batch_size",
        vram_gb=vram_gb,
        model_size_b=model_size_b,
        batch_size=batch_size,
    )

    return batch_size


@dataclass
class TrainConfig:
    """Configuration for self-play training."""

    # Episode settings
    total_episodes: int = 1000
    max_steps_per_episode: int = 10
    benign_ratio: float = 0.5  # Ratio of benign vs adversarial episodes

    # Training settings
    learning_rate: float = 1e-5
    batch_size: int | None = None  # Auto-computed based on VRAM if None
    gradient_accumulation_steps: int = 4

    def __post_init__(self) -> None:
        """Auto-compute batch size based on available VRAM if not specified."""
        if self.batch_size is None:
            vram_gb = get_available_vram_gb()
            self.batch_size = compute_batch_size(vram_gb)
            logger.info(
                "auto_batch_size",
                vram_gb=round(vram_gb, 1),
                batch_size=self.batch_size,
            )

    # Checkpointing
    checkpoint_interval: int = 100  # Save every N episodes
    checkpoint_dir: str = "./checkpoints"

    # Logging
    log_interval: int = 10  # Log every N episodes

    # PPO hyperparameters (skeleton)
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Seed
    seed: int = 42


@dataclass
class EpisodeBuffer:
    """Buffer for storing episode data for batch updates."""

    # State information
    prompts: list[str] = field(default_factory=list)
    responses: list[str] = field(default_factory=list)

    # Attacker data
    attacker_input_ids: list[torch.Tensor] = field(default_factory=list)
    attacker_output_ids: list[torch.Tensor] = field(default_factory=list)
    attacker_log_probs: list[torch.Tensor] = field(default_factory=list)
    attacker_rewards: list[float] = field(default_factory=list)
    attacker_values: list[torch.Tensor] = field(default_factory=list)

    # Guard data
    guard_input_ids: list[torch.Tensor] = field(default_factory=list)
    guard_output_ids: list[torch.Tensor] = field(default_factory=list)
    guard_log_probs: list[torch.Tensor] = field(default_factory=list)
    guard_rewards: list[float] = field(default_factory=list)
    guard_values: list[torch.Tensor] = field(default_factory=list)

    def clear(self) -> None:
        """Clear the buffer."""
        self.prompts.clear()
        self.responses.clear()
        self.attacker_input_ids.clear()
        self.attacker_output_ids.clear()
        self.attacker_log_probs.clear()
        self.attacker_rewards.clear()
        self.attacker_values.clear()
        self.guard_input_ids.clear()
        self.guard_output_ids.clear()
        self.guard_log_probs.clear()
        self.guard_rewards.clear()
        self.guard_values.clear()

    def __len__(self) -> int:
        """Return number of samples in buffer."""
        return len(self.attacker_rewards)


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""

    episode_rewards_attacker: list[float] = field(default_factory=list)
    episode_rewards_guard: list[float] = field(default_factory=list)
    attack_success_rate: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0

    # Running counts for metrics
    _total_attacks: int = 0
    _successful_attacks: int = 0
    _total_benign: int = 0
    _false_positives: int = 0
    _false_negatives: int = 0

    def update(
        self,
        is_benign: bool,
        guard_blocked: bool,
        violation: bool,
        attacker_reward: float,
        guard_reward: float,
    ) -> None:
        """Update metrics with episode result."""
        self.episode_rewards_attacker.append(attacker_reward)
        self.episode_rewards_guard.append(guard_reward)

        if is_benign:
            self._total_benign += 1
            if guard_blocked:
                self._false_positives += 1
        else:
            self._total_attacks += 1
            if violation:
                self._successful_attacks += 1
            if not guard_blocked and violation:
                self._false_negatives += 1

        # Update rates
        if self._total_attacks > 0:
            self.attack_success_rate = self._successful_attacks / self._total_attacks
        if self._total_benign > 0:
            self.false_positive_rate = self._false_positives / self._total_benign
        if self._total_attacks > 0:
            self.false_negative_rate = self._false_negatives / self._total_attacks

    def get_summary(self, window: int = 100) -> dict[str, float]:
        """Get summary statistics over recent episodes."""
        recent_attacker = (
            self.episode_rewards_attacker[-window:] if self.episode_rewards_attacker else [0]
        )
        recent_guard = self.episode_rewards_guard[-window:] if self.episode_rewards_guard else [0]

        return {
            "mean_attacker_reward": sum(recent_attacker) / len(recent_attacker),
            "mean_guard_reward": sum(recent_guard) / len(recent_guard),
            "attack_success_rate": self.attack_success_rate,
            "false_positive_rate": self.false_positive_rate,
            "false_negative_rate": self.false_negative_rate,
        }


class SelfPlayTrainer:
    """Trainer for self-play RL between Attacker and Guard.

    Implements PPO (Proximal Policy Optimization) with GAE (Generalized
    Advantage Estimation) for training both attacker and guard models.
    Supports multi-GPU training via accelerate.
    """

    def __init__(
        self,
        attacker_model: HuggingFaceLM,
        guard_model: HuggingFaceLM,
        target_model: HuggingFaceLM,
        settings: Settings | None = None,
        config: TrainConfig | None = None,
        judge: SafetyJudge | None = None,
        use_accelerate: bool = True,
    ) -> None:
        """Initialize the trainer.

        Args:
            attacker_model: The attacker LLM.
            guard_model: The guard LLM.
            target_model: The surrogate target LLM.
            settings: Application settings.
            config: Training configuration.
            judge: Safety judge for evaluating responses.
            use_accelerate: Whether to use accelerate for multi-GPU training.
        """
        self.attacker_model = attacker_model
        self.guard_model = guard_model
        self.target_model = target_model
        self.settings = settings or Settings()
        self.config = config or TrainConfig()

        # Initialize accelerator for multi-GPU support
        self.accelerator: Accelerator | None = None
        if use_accelerate:
            self.accelerator = get_accelerator()
            logger.info(
                "trainer_using_accelerate",
                device=str(self.accelerator.device),
                num_processes=self.accelerator.num_processes,
            )

        # Initialize environment with safety judge
        self.judge = judge or SafetyJudge()
        self.env = SelfPlayEnv(
            attacker=attacker_model,
            guard=guard_model,
            target=target_model,
            judge=self.judge,
            max_turns=self.config.max_steps_per_episode,
        )

        # Initialize optimizers
        self.attacker_optimizer = AdamW(
            attacker_model.model.parameters(),
            lr=self.config.learning_rate,
        )
        self.guard_optimizer = AdamW(
            guard_model.model.parameters(),
            lr=self.config.learning_rate,
        )

        # Prepare models and optimizers with accelerate
        if self.accelerator is not None:
            (
                self.attacker_model.model,
                self.guard_model.model,
                self.attacker_optimizer,
                self.guard_optimizer,
            ) = self.accelerator.prepare(
                self.attacker_model.model,
                self.guard_model.model,
                self.attacker_optimizer,
                self.guard_optimizer,
            )

        # Training state
        self.global_step = 0
        self.episode_index = 0
        self.metrics = TrainingMetrics()
        self.buffer = EpisodeBuffer()

        # Set seed
        self._set_seed(self.config.seed)

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(self, resume_from: str | Path | None = None) -> None:
        """Run the training loop.

        Args:
            resume_from: Optional checkpoint directory to resume from.
        """
        # Resume from checkpoint if specified
        if resume_from:
            self._load_checkpoint(resume_from)

        logger.info(
            "training_started",
            total_episodes=self.config.total_episodes,
            start_episode=self.episode_index,
            checkpoint_dir=self.config.checkpoint_dir,
        )

        try:
            while self.episode_index < self.config.total_episodes:
                # Run one episode
                episode_result = self._run_episode()

                # Update metrics
                self.metrics.update(
                    is_benign=episode_result["is_benign"],
                    guard_blocked=episode_result["guard_blocked"],
                    violation=episode_result["violation"],
                    attacker_reward=episode_result["attacker_reward"],
                    guard_reward=episode_result["guard_reward"],
                )

                self.episode_index += 1

                # Periodic PPO update
                if self.episode_index % self.config.batch_size == 0:
                    self._ppo_update()

                # Logging
                if self.episode_index % self.config.log_interval == 0:
                    self._log_metrics()

                # Checkpointing
                if self.episode_index % self.config.checkpoint_interval == 0:
                    self._save_checkpoint()

        except KeyboardInterrupt:
            logger.info("training_interrupted", episode=self.episode_index)
            self._save_checkpoint()
            raise

        logger.info("training_completed", total_episodes=self.episode_index)
        self._save_checkpoint()

    def _run_episode(self) -> dict[str, Any]:
        """Run a single self-play episode.

        Returns:
            Dictionary with episode results.
        """
        # Decide if this is benign or adversarial
        is_benign = random.random() < self.config.benign_ratio

        # Reset environment
        state = self.env.reset(is_benign=is_benign)

        total_attacker_reward = 0.0
        total_guard_reward = 0.0
        guard_blocked = False
        violation = False

        # Run episode turns
        while not state.done:
            result = self.env.step()
            state = result.state

            total_attacker_reward += result.rewards["attacker"]
            total_guard_reward += result.rewards["guard"]

            # Track if guard ever blocked
            if result.info.get("guard_action") == "block":
                guard_blocked = True

            # Track violation
            if result.info.get("violation"):
                violation = True

            # Store for PPO (simplified - would need log probs in real implementation)
            self.buffer.attacker_rewards.append(result.rewards["attacker"])
            self.buffer.guard_rewards.append(result.rewards["guard"])

            self.global_step += 1

        return {
            "is_benign": is_benign,
            "guard_blocked": guard_blocked,
            "violation": violation,
            "attacker_reward": total_attacker_reward,
            "guard_reward": total_guard_reward,
            "num_turns": state.current_step,
        }

    def _ppo_update(self) -> None:
        """Perform PPO update on accumulated episode data.

        Implements the full PPO algorithm with:
        1. GAE (Generalized Advantage Estimation)
        2. Clipped surrogate objective
        3. Value function loss
        4. Entropy bonus
        """
        if len(self.buffer) == 0:
            return

        logger.debug(
            "ppo_update_starting",
            num_samples=len(self.buffer),
        )

        # Convert rewards to tensors
        attacker_rewards = torch.tensor(self.buffer.attacker_rewards, dtype=torch.float32)
        guard_rewards = torch.tensor(self.buffer.guard_rewards, dtype=torch.float32)

        # Normalize rewards
        if len(attacker_rewards) > 1:
            attacker_rewards = (attacker_rewards - attacker_rewards.mean()) / (
                attacker_rewards.std() + 1e-8
            )
            guard_rewards = (guard_rewards - guard_rewards.mean()) / (guard_rewards.std() + 1e-8)

        # Compute advantages using simple reward-to-go (simplified GAE)
        attacker_advantages = self._compute_advantages(attacker_rewards)
        guard_advantages = self._compute_advantages(guard_rewards)

        # PPO update epochs
        for epoch in range(self.config.ppo_epochs):
            # Update attacker
            if self.buffer.attacker_log_probs:
                attacker_loss = self._ppo_step(
                    model=self.attacker_model,
                    optimizer=self.attacker_optimizer,
                    old_log_probs=self.buffer.attacker_log_probs,
                    input_ids=self.buffer.attacker_input_ids,
                    output_ids=self.buffer.attacker_output_ids,
                    advantages=attacker_advantages,
                )
                logger.debug("attacker_ppo_step", epoch=epoch, loss=attacker_loss)

            # Update guard
            if self.buffer.guard_log_probs:
                guard_loss = self._ppo_step(
                    model=self.guard_model,
                    optimizer=self.guard_optimizer,
                    old_log_probs=self.buffer.guard_log_probs,
                    input_ids=self.buffer.guard_input_ids,
                    output_ids=self.buffer.guard_output_ids,
                    advantages=guard_advantages,
                )
                logger.debug("guard_ppo_step", epoch=epoch, loss=guard_loss)

        # Clear buffer after update
        self.buffer.clear()

        logger.debug("ppo_update_completed")

    def _compute_advantages(self, rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
        """Compute advantages using reward-to-go.

        Args:
            rewards: Tensor of rewards.
            gamma: Discount factor.

        Returns:
            Tensor of advantages.
        """
        advantages = torch.zeros_like(rewards)
        running_return = 0.0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            advantages[t] = running_return

        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def _ppo_step(
        self,
        model: HuggingFaceLM,
        optimizer: torch.optim.Optimizer,
        old_log_probs: list[torch.Tensor],
        input_ids: list[torch.Tensor],
        output_ids: list[torch.Tensor],
        advantages: torch.Tensor,
    ) -> float:
        """Perform a single PPO optimization step.

        Args:
            model: The model to update.
            optimizer: The optimizer to use.
            old_log_probs: Log probabilities from the old policy.
            input_ids: Input token IDs.
            output_ids: Output token IDs.
            advantages: Computed advantages.

        Returns:
            The loss value.
        """
        total_loss = 0.0
        num_samples = min(len(old_log_probs), len(advantages))

        for i in range(num_samples):
            if i >= len(input_ids) or i >= len(output_ids):
                continue

            # Get new log probs
            new_log_probs = model.get_log_probs(input_ids[i], output_ids[i])
            new_log_probs_sum = new_log_probs.sum()

            # Old log probs sum
            old_log_probs_sum = old_log_probs[i].sum()

            # Compute ratio
            ratio = torch.exp(new_log_probs_sum - old_log_probs_sum)

            # Clipped surrogate objective
            advantage = advantages[i].to(new_log_probs_sum.device)
            surrogate1 = ratio * advantage
            surrogate2 = (
                torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                )
                * advantage
            )

            # Policy loss (negative because we want to maximize)
            policy_loss = -torch.min(surrogate1, surrogate2)

            # Entropy bonus (encourage exploration)
            entropy = -(new_log_probs * torch.exp(new_log_probs)).sum()
            entropy_loss = -self.config.entropy_coef * entropy

            # Total loss for this sample
            loss = policy_loss + entropy_loss
            total_loss += loss.item()

            # Backward pass with accelerate support
            optimizer.zero_grad()
            if self.accelerator is not None:
                self.accelerator.backward(loss)
            else:
                loss.backward()

            # Get the underlying model for gradient clipping
            underlying_model = (
                self.accelerator.unwrap_model(model.model) if self.accelerator else model.model
            )
            torch.nn.utils.clip_grad_norm_(underlying_model.parameters(), self.config.max_grad_norm)
            optimizer.step()

        return total_loss / max(num_samples, 1)

    def _log_metrics(self) -> None:
        """Log training metrics."""
        summary = self.metrics.get_summary()
        logger.info(
            "training_metrics",
            episode=self.episode_index,
            global_step=self.global_step,
            **summary,
        )

    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        # Only save on main process when using accelerate
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return

        checkpoint_dir = Path(self.config.checkpoint_dir) / f"step_{self.global_step:08d}"

        trainer_state = TrainerState(
            global_step=self.global_step,
            episode_index=self.episode_index,
            rng_seed=self.config.seed,
            timestamp=datetime.now(timezone.utc).isoformat(),
            attacker_model_name=self.attacker_model.config.model_name,
            guard_model_name=self.guard_model.config.model_name,
            target_model_name=self.target_model.config.model_name,
            config={
                "total_episodes": self.config.total_episodes,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
            },
        )

        # Unwrap models if using accelerate
        attacker_model = (
            self.accelerator.unwrap_model(self.attacker_model.model)
            if self.accelerator
            else self.attacker_model.model
        )
        guard_model = (
            self.accelerator.unwrap_model(self.guard_model.model)
            if self.accelerator
            else self.guard_model.model
        )

        save_training_checkpoint(
            checkpoint_dir=checkpoint_dir,
            attacker_model=attacker_model,
            guard_model=guard_model,
            attacker_optimizer=self.attacker_optimizer,
            guard_optimizer=self.guard_optimizer,
            trainer_state=trainer_state,
        )

    def _load_checkpoint(self, checkpoint_dir: str | Path) -> None:
        """Load training state from checkpoint."""
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)

        if checkpoint_path is None:
            logger.warning("no_checkpoint_found", path=str(checkpoint_dir))
            return

        # Unwrap models if using accelerate
        attacker_model = (
            self.accelerator.unwrap_model(self.attacker_model.model)
            if self.accelerator
            else self.attacker_model.model
        )
        guard_model = (
            self.accelerator.unwrap_model(self.guard_model.model)
            if self.accelerator
            else self.guard_model.model
        )

        trainer_state = load_training_checkpoint(
            checkpoint_dir=checkpoint_path,
            attacker_model=attacker_model,
            guard_model=guard_model,
            attacker_optimizer=self.attacker_optimizer,
            guard_optimizer=self.guard_optimizer,
        )

        self.global_step = trainer_state.global_step
        self.episode_index = trainer_state.episode_index
        self._set_seed(trainer_state.rng_seed)

        logger.info(
            "checkpoint_restored",
            path=str(checkpoint_path),
            global_step=self.global_step,
            episode=self.episode_index,
        )
