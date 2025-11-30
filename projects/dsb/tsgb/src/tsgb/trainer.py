"""Self-play RL trainer for Stage 1 training.

This module provides PPO-style self-play training between Attacker and Guard models.
Uses proper PPO implementation with GAE advantage estimation.
Supports multi-GPU training via accelerate.
"""

import random
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW

from tsgb.checkpoint import (
    TrainerState,
    find_latest_checkpoint,
    load_training_checkpoint,
    save_training_checkpoint,
)
from tsgb.envs import SafetyJudge, SelfPlayEnv
from tsgb.logging import get_logger
from tsgb.models import GenerationOutput, HuggingFaceLM, LLMRole, get_accelerator
from tsgb.settings import Settings

logger = get_logger(__name__)


def get_available_vram_gb() -> float:
    """Get currently free VRAM across all GPUs in GB.

    Returns:
        Total free VRAM in GB, or 0 if no GPU available.
    """
    if not torch.cuda.is_available():
        return 0.0

    total_free_vram = 0.0
    for i in range(torch.cuda.device_count()):
        try:
            free_mem, _ = torch.cuda.mem_get_info(i)
            total_free_vram += free_mem / (1024**3)
        except Exception:
            props = torch.cuda.get_device_properties(i)
            total_free_vram += props.total_memory / (1024**3)

    return total_free_vram


def compute_batch_size(vram_gb: float, model_size_b: float = 3.0) -> int:
    """Compute optimal batch size based on available VRAM.

    Uses a GPU-aware heuristic that scales with the number of devices and
    available free memory to better saturate multi-GPU boxes.

    Args:
        vram_gb: Currently free VRAM in GB across all GPUs.
        model_size_b: Model size in billions of parameters.

    Returns:
        Recommended batch size.
    """
    if vram_gb <= 0:
        return 1

    gpu_count = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1

    # Use 75% of available VRAM for better GPU utilization (increased from 50%)
    available_for_batch = vram_gb * 0.75

    # Activation memory per sample scales down when sharded across GPUs.
    # Reduced memory estimate with gradient checkpointing and mixed precision
    memory_per_sample = (4.0 * (model_size_b / 3.0)) / gpu_count

    batch_size = int(available_for_batch / memory_per_sample)

    # Allow larger batch sizes for high-VRAM setups (e.g., dual RTX 8000 = 96GB)
    # Max batch size scales with total VRAM: 256 for 48GB+, 512 for 80GB+
    max_batch = 128 if vram_gb < 48 else (256 if vram_gb < 80 else 512)
    batch_size = max(1, min(batch_size, max_batch))

    logger.info(
        "computed_batch_size",
        vram_gb=round(vram_gb, 2),
        model_size_b=model_size_b,
        gpu_count=gpu_count,
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
    episode_batch_size: int | None = None  # How many episodes to run concurrently
    gradient_accumulation_steps: int = 4
    use_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = True

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
        if self.episode_batch_size is None:
            self.episode_batch_size = self.batch_size

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
        trackio_run: Any | None = None,
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
            trackio_run: Optional Trackio run for logging metrics.
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
            self.accelerator = get_accelerator(
                mixed_precision="fp16" if self.config.use_mixed_precision else None,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            )
            logger.info(
                "trainer_using_accelerate",
                device=str(self.accelerator.device),
                num_processes=self.accelerator.num_processes,
            )

        self.is_main_process = self.accelerator.is_main_process if self.accelerator else True
        self.trackio_run = trackio_run if self.is_main_process else None
        self.episode_batch_size = max(1, self.config.episode_batch_size or 1)

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

    def _device_for_training(self) -> torch.device:
        """Resolve the device for training computations."""
        if self.accelerator is not None:
            return self.accelerator.device
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

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
                prev_episode_index = self.episode_index
                remaining = self.config.total_episodes - self.episode_index
                batch_size = min(self.episode_batch_size, remaining)

                batch_results = self._run_episode_batch(batch_size)

                for episode_result in batch_results:
                    self.metrics.update(
                        is_benign=episode_result["is_benign"],
                        guard_blocked=episode_result["guard_blocked"],
                        violation=episode_result["violation"],
                        attacker_reward=episode_result["attacker_reward"],
                        guard_reward=episode_result["guard_reward"],
                    )

                self.episode_index += len(batch_results)

                # Periodic PPO update when buffer is large enough
                if len(self.buffer) >= (self.config.batch_size or batch_size):
                    self._ppo_update()

                # Logging when crossing interval boundaries
                if (prev_episode_index // self.config.log_interval) != (
                    self.episode_index // self.config.log_interval
                ):
                    self._log_metrics()

                # Checkpointing when crossing interval boundaries
                if (prev_episode_index // self.config.checkpoint_interval) != (
                    self.episode_index // self.config.checkpoint_interval
                ):
                    self._save_checkpoint()

        except KeyboardInterrupt:
            logger.info("training_interrupted", episode=self.episode_index)
            self._save_checkpoint()
            raise

        if len(self.buffer) > 0:
            self._ppo_update()

        logger.info("training_completed", total_episodes=self.episode_index)
        self._save_checkpoint()

    def _run_episode(self) -> dict[str, Any]:
        """Run a single self-play episode (wrapper for batch executor)."""
        return self._run_episode_batch(1)[0]

    def _run_episode_batch(self, batch_size: int) -> list[dict[str, Any]]:
        """Run multiple episodes concurrently with batched generation."""
        envs = [
            SelfPlayEnv(
                attacker=self.attacker_model,
                guard=self.guard_model,
                target=self.target_model,
                judge=self.judge,
                max_turns=self.config.max_steps_per_episode,
            )
            for _ in range(batch_size)
        ]

        states = [env.reset(is_benign=random.random() < self.config.benign_ratio) for env in envs]
        done = [False] * batch_size

        total_attacker_rewards = [0.0] * batch_size
        total_guard_rewards = [0.0] * batch_size
        guard_blocked_flags = [False] * batch_size
        violation_flags = [False] * batch_size

        episode_results: list[dict[str, Any]] = [
            {
                "is_benign": state.is_benign,
                "guard_blocked": False,
                "violation": False,
                "attacker_reward": 0.0,
                "guard_reward": 0.0,
                "num_turns": 0,
            }
            for state in states
        ]

        while not all(done):
            active_indices = [i for i, flag in enumerate(done) if not flag]

            # 1) Attacker generates messages (batched)
            attacker_prompts = [env._build_attacker_prompt(states[i]) for i in active_indices]
            attacker_outputs = self.attacker_model.generate_batch(
                attacker_prompts,
                return_logits=False,
            )

            # Apply attacker outputs to state
            for idx, output in zip(active_indices, attacker_outputs):
                states[idx].add_turn("attacker", output.text, scenario_id=states[idx].scenario_id)

            # 2) Guard decides actions (batched)
            guard_prompts = [env._build_guard_prompt(states[i]) for i in active_indices]
            guard_outputs = self.guard_model.generate_batch(
                guard_prompts,
                return_logits=False,
            )
            guard_actions = [
                envs[idx]._parse_guard_action(output.text)
                for idx, output in zip(active_indices, guard_outputs)
            ]

            for idx, action in zip(active_indices, guard_actions):
                states[idx].add_turn("guard", f"ACTION: {action}")
                guard_blocked_flags[idx] = guard_blocked_flags[idx] or action == "block"

            # 3) Target responses for allowed actions (batched only on allowed)
            allow_indices = [
                idx for idx, action in zip(active_indices, guard_actions) if action == "allow"
            ]
            target_outputs_map: dict[int, GenerationOutput] = {}
            if allow_indices:
                target_prompts = [
                    envs[idx]._build_target_prompt(states[idx]) for idx in allow_indices
                ]
                target_outputs = self.target_model.generate_batch(
                    target_prompts, return_logits=False
                )
                for idx, output in zip(allow_indices, target_outputs):
                    target_outputs_map[idx] = output
                    states[idx].add_turn("target", output.text)

            # 4) Judge and rewards
            for idx, action in zip(active_indices, guard_actions):
                if action == "allow":
                    target_response = target_outputs_map[idx].text
                    is_violation = self.judge.judge(
                        target_response,
                        states[idx].is_benign,
                        states[idx].scenario_id,
                    )
                    states[idx].violation = is_violation
                    violation_flags[idx] = violation_flags[idx] or is_violation
                else:
                    target_response = ""
                    states[idx].violation = False

                rewards = envs[idx]._compute_rewards(
                    state=states[idx],
                    guard_action=action,
                    violation=states[idx].violation,
                )

                total_attacker_rewards[idx] += rewards["attacker"]
                total_guard_rewards[idx] += rewards["guard"]

                # Store PPO buffers with log probs
                attacker_output = attacker_outputs[active_indices.index(idx)]
                guard_output = guard_outputs[active_indices.index(idx)]

                attacker_log_probs = (
                    self.attacker_model.get_log_probs(
                        attacker_output.input_ids.to(self._device_for_training()),
                        attacker_output.output_ids.to(self._device_for_training()),
                    )
                    .detach()
                    .cpu()
                )
                guard_log_probs = (
                    self.guard_model.get_log_probs(
                        guard_output.input_ids.to(self._device_for_training()),
                        guard_output.output_ids.to(self._device_for_training()),
                    )
                    .detach()
                    .cpu()
                )

                self.buffer.attacker_input_ids.append(attacker_output.input_ids.detach().cpu())
                self.buffer.attacker_output_ids.append(attacker_output.output_ids.detach().cpu())
                self.buffer.attacker_log_probs.append(attacker_log_probs)
                self.buffer.attacker_rewards.append(rewards["attacker"])

                self.buffer.guard_input_ids.append(guard_output.input_ids.detach().cpu())
                self.buffer.guard_output_ids.append(guard_output.output_ids.detach().cpu())
                self.buffer.guard_log_probs.append(guard_log_probs)
                self.buffer.guard_rewards.append(rewards["guard"])

                # Episode termination check
                if (
                    states[idx].current_step >= self.config.max_steps_per_episode
                    or states[idx].violation
                ):
                    states[idx].done = True

                done[idx] = states[idx].done
                episode_results[idx] = {
                    "is_benign": states[idx].is_benign,
                    "guard_blocked": guard_blocked_flags[idx],
                    "violation": violation_flags[idx],
                    "attacker_reward": total_attacker_rewards[idx],
                    "guard_reward": total_guard_rewards[idx],
                    "num_turns": states[idx].current_step,
                }

            # Increase global step by number of active episodes progressed
            self.global_step += len(active_indices)

        return episode_results

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
        """Perform a batched PPO optimization step."""
        if not input_ids or not output_ids or not old_log_probs:
            return 0.0

        num_samples = min(len(old_log_probs), len(advantages), len(input_ids), len(output_ids))
        device = self._device_for_training()

        # Pad sequences to batch them
        pad_token_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id or 0
        padded_input_ids = pad_sequence(
            [ids.squeeze(0) for ids in input_ids[:num_samples]],
            batch_first=True,
            padding_value=pad_token_id,
        ).to(device)
        padded_output_ids = pad_sequence(
            [ids.squeeze(0) for ids in output_ids[:num_samples]],
            batch_first=True,
            padding_value=pad_token_id,
        ).to(device)

        output_lengths = torch.tensor(
            [ids.shape[1] for ids in output_ids[:num_samples]], device=device, dtype=torch.long
        )
        output_mask = (
            torch.arange(padded_output_ids.shape[1], device=device)[None, :]
            < output_lengths[:, None]
        )

        padded_old_log_probs = pad_sequence(
            [
                probs.squeeze(0) if probs.dim() > 1 else probs
                for probs in old_log_probs[:num_samples]
            ],
            batch_first=True,
            padding_value=0.0,
        ).to(device)

        advantages_batch = advantages[:num_samples].to(device)

        amp_context = (
            (
                self.accelerator.autocast()
                if self.accelerator
                else torch.autocast("cuda", dtype=torch.float16)
            )
            if (self.config.use_mixed_precision and torch.cuda.is_available())
            else nullcontext()
        )

        optimizer.zero_grad()
        with amp_context:
            new_log_probs = model.get_log_probs(padded_input_ids, padded_output_ids)
            masked_new_log_probs = new_log_probs * output_mask
            masked_old_log_probs = padded_old_log_probs * output_mask

            new_log_probs_sum = masked_new_log_probs.sum(dim=1)
            old_log_probs_sum = masked_old_log_probs.sum(dim=1)

            ratio = torch.exp(new_log_probs_sum - old_log_probs_sum)
            surrogate1 = ratio * advantages_batch
            surrogate2 = (
                torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                )
                * advantages_batch
            )
            policy_loss = -torch.min(surrogate1, surrogate2)

            entropy = -(masked_new_log_probs.exp() * masked_new_log_probs).sum(dim=1)
            entropy_loss = -self.config.entropy_coef * entropy

            losses = policy_loss + entropy_loss
            loss = losses.mean()

        if self.accelerator is not None:
            with self.accelerator.accumulate(model.model):
                self.accelerator.backward(loss)
                underlying_model = self.accelerator.unwrap_model(model.model)
                torch.nn.utils.clip_grad_norm_(
                    underlying_model.parameters(),
                    self.config.max_grad_norm,
                )
                optimizer.step()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), self.config.max_grad_norm)
            optimizer.step()

        return float(loss.detach().cpu())

    def _log_metrics(self) -> None:
        """Log training metrics."""
        summary = self.metrics.get_summary()
        logger.info(
            "training_metrics",
            episode=self.episode_index,
            global_step=self.global_step,
            **summary,
        )

        if self.trackio_run is not None:
            try:
                self.trackio_run.log(
                    {
                        "training/episode": self.episode_index,
                        "training/global_step": self.global_step,
                        **summary,
                    },
                    step=self.global_step,
                )
            except Exception as e:  # pragma: no cover - best effort logging
                logger.debug("trackio_log_failed", error=str(e))

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
