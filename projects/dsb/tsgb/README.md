# TSGB: Train Small, Guard Big

LLM jailbreak attack/defense self-play framework with Vast.ai interruptible orchestration for research experiments.

## Concept

This project implements a two-stage approach for training robust LLM safety guards:

**Stage 1: Self-Play Training**
- Train an Attacker LLM (A) and Guard LLM (G) via self-play RL on a small surrogate target (T_s, e.g., 3B open-source model)
- The attacker learns to generate adversarial prompts
- The guard learns to detect and block jailbreak attempts

**Stage 2: Transfer Evaluation**
- Deploy the trained Guard (G) in front of black-box LLMs (T_b) like GPT-4, Claude, etc.
- Evaluate Attack Success Rate (ASR), False Positive Rate (FPR), and False Negative Rate (FNR)

## Safety Notice

This repository provides a **research scaffold only**. The default code does NOT generate actual harmful jailbreak content. All attack scenarios are abstracted and identified only by scenario IDs. Researchers must follow responsible AI safety guidelines when extending this framework.

## Installation

```bash
# Using uv (recommended)
uv sync

# Install with dev dependencies
uv sync --all-extras
```

## Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Required environment variables:
- `VAST_API_KEY`: Your Vast.ai API key
- `RCLONE_WEBDAV_URL`: WebDAV URL for checkpoint storage (optional)
- `RCLONE_WEBDAV_USER`: WebDAV username (optional)
- `RCLONE_WEBDAV_PASS`: WebDAV password (optional)

## Usage

### Local Training (Stage 1)

Run self-play RL training locally:

```bash
uv run tsgb train-local --episodes 100 --checkpoint-interval 10
```

### Local Evaluation (Stage 2)

Evaluate the trained guard with a dummy black-box LLM:

```bash
uv run tsgb eval-local --checkpoint-path ./checkpoints/latest
```

### Vast.ai Orchestration

For training on Vast.ai interruptible instances:

1. Configure your `.env` with Vast.ai API key and WebDAV storage settings

2. Start the manager to provision and monitor instances:
```bash
# Single check
uv run tsgb manager run

# Continuous monitoring (every 60 seconds)
uv run tsgb manager loop --interval 60
```

3. The manager will:
   - Find cost-effective GPU instances matching your requirements
   - Provision instances with automatic environment setup
   - Mount WebDAV storage for checkpoint persistence
   - Resume training from the latest checkpoint on instance restart

### Worker (on Vast.ai instance)

The worker runs automatically on provisioned instances:

```bash
uv run tsgb worker run --resume-path /mnt/tsgb/checkpoints
```

## Architecture

### Interruptible Training Flow

```
[Manager (stable server)]
    |
    +-- Monitors Vast.ai instance state
    +-- Provisions new instances when needed
    +-- Injects onstart script
    
[Worker (Vast.ai instance)]
    |
    +-- Mounts WebDAV via rclone
    +-- Loads latest checkpoint
    +-- Runs self-play training
    +-- Saves checkpoints periodically
    +-- Handles SIGINT/SIGTERM gracefully
```

### Checkpoint Storage

Checkpoints are stored using safetensors format:
- `attacker.safetensors`: Attacker model weights
- `guard.safetensors`: Guard model weights  
- `optim_attacker.safetensors`: Attacker optimizer state (tensors)
- `optim_guard.safetensors`: Guard optimizer state (tensors)
- `trainer_state.json`: Training metadata (step, episode, RNG seed, timestamp)

## Project Structure

```
tsgb/
├── pyproject.toml       # Project configuration and dependencies
├── README.md            # This file
├── .env.example         # Example environment variables
└── src/tsgb/
    ├── __init__.py      # Package initialization
    ├── settings.py      # Configuration with pydantic-settings
    ├── logging.py       # Structured logging setup
    ├── models.py        # LLM wrapper classes
    ├── envs.py          # Self-play environment
    ├── checkpoint.py    # Checkpoint save/load utilities
    ├── trainer.py       # RL trainer skeleton
    ├── eval.py          # Stage 2 evaluation
    ├── vast_api.py      # Vast.ai API client
    ├── manager.py       # Instance manager
    ├── worker.py        # Training worker
    └── cli.py           # Typer CLI application
```

## Extending the Framework

### Custom Safety Judge

Replace `SimpleJudge` in `envs.py` with your own classifier:

```python
class MyJudge:
    def judge(self, response: str, is_benign: bool) -> bool:
        # Your safety classification logic
        return is_violation
```

### Black-Box LLM Integration

Implement the `BlackBoxLLM` protocol in `eval.py`:

```python
class GPT4LLM:
    def generate(self, prompt: str) -> str:
        # Call OpenAI API
        return response
```

## License

MIT License - See LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{tsgb2024,
  title = {TSGB: Train Small, Guard Big},
  year = {2024},
  url = {https://github.com/your-org/tsgb}
}
```
