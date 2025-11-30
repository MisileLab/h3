"""Vast.ai instance manager for orchestrating training workers.

Manages the lifecycle of Vast.ai GPU instances for distributed training:
- Monitors instance health
- Provisions new instances when needed
- Injects onstart scripts for automatic environment setup
"""

import base64
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tsgb.logging import get_logger
from tsgb.settings import Settings, get_settings
from tsgb.vast_api import Instance, VastAPIClient

logger = get_logger(__name__)


@dataclass
class ManagerState:
    """Persistent state of the manager."""

    instance_id: int | None = None
    status: str = "unknown"
    last_check_at: str = ""
    last_provision_at: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "instance_id": self.instance_id,
            "status": self.status,
            "last_check_at": self.last_check_at,
            "last_provision_at": self.last_provision_at,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManagerState":
        """Create from dictionary."""
        return cls(
            instance_id=data.get("instance_id"),
            status=data.get("status", "unknown"),
            last_check_at=data.get("last_check_at", ""),
            last_provision_at=data.get("last_provision_at", ""),
            created_at=data.get("created_at", ""),
        )


def generate_env_content(settings: Settings) -> str:
    """Generate .env file content from current settings.

    Args:
        settings: Application settings.

    Returns:
        Content for .env file.
    """
    lines = []

    # Vast.ai API
    if settings.vast_api_key:
        lines.append(f"VAST_API_KEY={settings.vast_api_key}")

    # WebDAV / rclone settings
    if settings.rclone_webdav_url:
        lines.append(f"RCLONE_WEBDAV_URL={settings.rclone_webdav_url}")
    if settings.rclone_webdav_user:
        lines.append(f"RCLONE_WEBDAV_USER={settings.rclone_webdav_user}")
    if settings.rclone_webdav_pass:
        lines.append(f"RCLONE_WEBDAV_PASS={settings.rclone_webdav_pass}")
    lines.append(f"RCLONE_REMOTE_NAME={settings.rclone_remote_name}")
    lines.append(f"RCLONE_MOUNTPOINT={settings.rclone_mountpoint}")

    # Checkpoint directories
    lines.append(f"CHECKPOINT_DIR={settings.checkpoint_dir}")
    lines.append(f"LOCAL_CHECKPOINT_DIR={settings.local_checkpoint_dir}")

    # Vast.ai instance filter defaults
    if settings.vast_gpu_name:
        lines.append(f"VAST_GPU_NAME={settings.vast_gpu_name}")
    lines.append(f"VAST_INSTANCE_TYPE={settings.vast_instance_type}")
    lines.append(f"VAST_MIN_VRAM_GB={settings.vast_min_vram_gb}")
    lines.append(f"VAST_MAX_PRICE={settings.vast_max_price}")

    # Training defaults
    lines.append(f"DEFAULT_MODEL_NAME={settings.default_model_name}")

    # API keys for black-box LLM providers
    if settings.openai_api_key:
        lines.append(f"OPENAI_API_KEY={settings.openai_api_key}")
    if settings.anthropic_api_key:
        lines.append(f"ANTHROPIC_API_KEY={settings.anthropic_api_key}")
    if settings.google_api_key:
        lines.append(f"GOOGLE_API_KEY={settings.google_api_key}")

    # Logging
    lines.append(f"LOG_MODE={settings.log_mode}")

    return "\n".join(lines) + "\n"


def generate_onstart_script(
    settings: Settings,
    repo_url: str = "https://gith.misile.xyz/h3.git:/projects/dsb/tsgb.git",
    branch: str = "main",
) -> str:
    """Generate the onstart script for Vast.ai instances.

    This script:
    1. Updates the system and installs dependencies
    2. Installs uv package manager
    3. Clones or updates the repository
    4. Creates .env file from base64-encoded settings
    5. Syncs dependencies with uv
    6. Configures and mounts WebDAV storage via rclone
    7. Starts the training worker

    NOTE: Sensitive credentials are passed via base64-encoded .env content
    in an environment variable, then decoded and written to .env file.

    Args:
        settings: Application settings.
        repo_url: Git repository URL.
        branch: Git branch to checkout.

    Returns:
        Bash script string for onstart.
    """
    # The script uses environment variables that should be set in Vast.ai instance config
    script = f"""#!/bin/bash
set -e

echo "=== TSGB Worker Startup Script ==="
echo "Started at: $(date -Iseconds)"

# Update system and install dependencies
apt-get update
apt-get install -y git curl fuse rclone

# Install uv package manager
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
export PATH="$HOME/.local/bin:$PATH"

# Clone or update repository
REPO_DIR="/workspace/tsgb"
if [ -d "$REPO_DIR" ]; then
    echo "Updating existing repository..."
    cd "$REPO_DIR"
    git fetch origin
    git checkout {branch}
    git pull origin {branch}
else
    echo "Cloning repository..."
    git clone --branch {branch} {repo_url} "$REPO_DIR"
    cd "$REPO_DIR"
fi

# Create .env file from base64-encoded environment variable
if [ -n "$TSGB_ENV_B64" ]; then
    echo "Creating .env file from encoded settings..."
    echo "$TSGB_ENV_B64" | base64 -d > "$REPO_DIR/.env"
    echo ".env file created successfully"
else
    echo "Warning: No TSGB_ENV_B64 found, .env file not created"
fi

# Sync dependencies
echo "Syncing dependencies with uv..."
uv sync

# Configure rclone for WebDAV if credentials are provided
MOUNTPOINT="${{RCLONE_MOUNTPOINT:-{settings.rclone_mountpoint}}}"
REMOTE_NAME="${{RCLONE_REMOTE_NAME:-{settings.rclone_remote_name}}}"

if [ -n "$RCLONE_WEBDAV_URL" ] && [ -n "$RCLONE_WEBDAV_USER" ] && [ -n "$RCLONE_WEBDAV_PASS" ]; then
    echo "Configuring rclone WebDAV remote..."
    
    # Create rclone config
    rclone config create "$REMOTE_NAME" webdav \\
        url "$RCLONE_WEBDAV_URL" \\
        vendor other \\
        user "$RCLONE_WEBDAV_USER" \\
        pass "$(rclone obscure "$RCLONE_WEBDAV_PASS")"
    
    # Create mount directory
    mkdir -p "$MOUNTPOINT"
    
    # Mount WebDAV in background with caching
    echo "Mounting WebDAV storage..."
    rclone mount "$REMOTE_NAME:" "$MOUNTPOINT" \\
        --vfs-cache-mode writes \\
        --dir-cache-time 5s \\
        --vfs-write-back 5s \\
        --allow-non-empty \\
        --daemon
    
    # Wait for mount to be ready
    echo "Waiting for mount to be ready..."
    for i in {{1..30}}; do
        if mountpoint -q "$MOUNTPOINT"; then
            echo "Mount ready!"
            break
        fi
        sleep 1
    done
    
    if ! mountpoint -q "$MOUNTPOINT"; then
        echo "Warning: Mount may not be ready, continuing anyway..."
    fi
    
    CHECKPOINT_PATH="$MOUNTPOINT/checkpoints"
else
    echo "No WebDAV credentials found, using local storage..."
    CHECKPOINT_PATH="./checkpoints"
fi

# Ensure checkpoint directory exists
mkdir -p "$CHECKPOINT_PATH"

# Start the training worker
echo "Starting training worker..."
echo "Checkpoint path: $CHECKPOINT_PATH"

cd "$REPO_DIR"
uv run tsgb worker run --resume-path "$CHECKPOINT_PATH"

echo "Worker exited at: $(date -Iseconds)"
"""
    return script


class InstanceManager:
    """Manages Vast.ai instance lifecycle for training workers."""

    def __init__(
        self,
        settings: Settings | None = None,
        state_file: str | Path = "manager_state.json",
    ) -> None:
        """Initialize the instance manager.

        Args:
            settings: Application settings.
            state_file: Path to persist manager state.
        """
        self.settings = settings or get_settings()
        self.state_file = Path(state_file)
        self.client = VastAPIClient.from_settings(self.settings)
        self.state = self._load_state()

    def _load_state(self) -> ManagerState:
        """Load manager state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                    logger.info("state_loaded", path=str(self.state_file))
                    return ManagerState.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("state_load_failed", error=str(e))

        return ManagerState(created_at=datetime.now(timezone.utc).isoformat())

    def _save_state(self) -> None:
        """Save manager state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)
        logger.debug("state_saved", path=str(self.state_file))

    def check_instance_status(self) -> Instance | None:
        """Check the status of the current instance.

        Returns:
            Instance if found and tracked, None otherwise.
        """
        if self.state.instance_id is None:
            logger.info("no_tracked_instance")
            return None

        try:
            instance = self.client.get_instance(self.state.instance_id)
            self.state.status = instance.status
            self.state.last_check_at = datetime.now(timezone.utc).isoformat()
            self._save_state()

            logger.info(
                "instance_status",
                instance_id=instance.id,
                status=instance.status,
                actual_status=instance.actual_status,
            )

            return instance

        except Exception as e:
            logger.warning(
                "instance_check_failed", instance_id=self.state.instance_id, error=str(e)
            )
            # If the instance is gone (e.g., outbid), clear tracking so we can reprovision
            self.state.instance_id = None
            self.state.status = "error"
            self._save_state()
            return None

    def find_best_offer(self) -> Any | None:
        """Find the best available offer matching requirements.

        Offers are sorted by DLPerf per dollar (best value first).

        Returns:
            Best offer if found, None otherwise.
        """
        offers = self.client.list_offers(
            min_vram_gb=self.settings.vast_min_vram_gb,
            max_price=self.settings.vast_max_price,
            verified=True,  # Prefer verified machines
            instance_type=self.settings.vast_instance_type,
            order_by="dlperf_per_dphtotal",
            order_dir="desc",  # Highest value first
            limit=10,
        )

        if not offers:
            logger.warning(
                "no_offers_found",
                min_vram=self.settings.vast_min_vram_gb,
                max_price=self.settings.vast_max_price,
            )
            return None

        # Return the best value offer (first in list, sorted by dlperf/$)
        best = offers[0]
        logger.info(
            "best_offer_found",
            offer_id=best.id,
            gpu_name=best.gpu_name,
            total_vram_gb=best.total_vram_gb,
            num_gpus=best.num_gpus,
            price=best.price_per_hour,
            dlperf=best.dlperf,
            dlperf_per_dollar=best.dlperf_per_dphtotal,
            min_bid=best.min_bid,
        )

        return best

    def provision_instance(
        self,
        repo_url: str = "https://gith.misile.xyz/h3.git:/projects/dsb/tsgb.git",
        branch: str = "main",
        disk_gb: int = 50,
        image: str = "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel",
    ) -> Instance | None:
        """Provision a new Vast.ai instance.

        Args:
            repo_url: Git repository URL.
            branch: Git branch.
            disk_gb: Disk space in GB.
            image: Docker image.

        Returns:
            Created instance if successful, None otherwise.
        """
        # Find best offer
        offer = self.find_best_offer()
        if offer is None:
            return None

        # Generate onstart script
        onstart = generate_onstart_script(
            settings=self.settings,
            repo_url=repo_url,
            branch=branch,
        )

        # Build environment variables for the instance
        env: dict[str, str] = {}

        # Generate .env content and encode as base64
        env_content = generate_env_content(self.settings)
        env_b64 = base64.b64encode(env_content.encode()).decode()
        env["TSGB_ENV_B64"] = env_b64

        # Also pass rclone credentials as env vars for the onstart script
        if self.settings.rclone_webdav_url:
            env["RCLONE_WEBDAV_URL"] = self.settings.rclone_webdav_url
        if self.settings.rclone_webdav_user:
            env["RCLONE_WEBDAV_USER"] = self.settings.rclone_webdav_user
        if self.settings.rclone_webdav_pass:
            env["RCLONE_WEBDAV_PASS"] = self.settings.rclone_webdav_pass
        env["RCLONE_REMOTE_NAME"] = self.settings.rclone_remote_name
        env["RCLONE_MOUNTPOINT"] = self.settings.rclone_mountpoint

        try:
            bid_price = None
            if self.settings.vast_instance_type == "bid":
                if offer.min_bid is not None:
                    bid_price = round(offer.min_bid * 1.02, 5)  # small cushion above min bid

            instance = self.client.create_instance(
                offer_id=offer.id,
                image=image,
                disk_gb=disk_gb,
                onstart=onstart,
                label="tsgb-worker",
                env=env,
                bid_price=bid_price,
            )

            # Update state
            self.state.instance_id = instance.id
            self.state.status = instance.status
            self.state.last_provision_at = datetime.now(timezone.utc).isoformat()
            self._save_state()

            logger.info(
                "instance_provisioned",
                instance_id=instance.id,
                offer_id=offer.id,
                gpu_name=offer.gpu_name,
                price=offer.price_per_hour,
            )

            return instance

        except Exception as e:
            logger.error("provision_failed", error=str(e))
            return None

    def ensure_worker_running(
        self,
        repo_url: str = "https://gith.misile.xyz/h3.git:/projects/dsb/tsgb.git",
        branch: str = "main",
    ) -> Instance | None:
        """Ensure a worker instance is running.

        Checks current instance status and provisions a new one if needed.

        Args:
            repo_url: Git repository URL.
            branch: Git branch.

        Returns:
            Running instance if available, None otherwise.
        """
        # Check current instance
        instance = self.check_instance_status()

        if instance is not None:
            if instance.is_running:
                logger.info("worker_running", instance_id=instance.id)
                return instance

            if instance.is_dead:
                logger.info("worker_dead", instance_id=instance.id, status=instance.status)
                # Clear the dead instance from state
                self.state.instance_id = None
                self.state.status = "dead"
                self._save_state()
                return None
            else:
                # Instance exists but not running (e.g., loading)
                logger.info("worker_starting", instance_id=instance.id, status=instance.status)
                return instance

        # No running instance, provision a new one
        logger.info("provisioning_new_worker")
        return self.provision_instance(repo_url=repo_url, branch=branch)

    def destroy_worker(self) -> bool:
        """Destroy the current worker instance.

        Returns:
            True if destroyed successfully.
        """
        if self.state.instance_id is None:
            logger.info("no_instance_to_destroy")
            return True

        try:
            self.client.destroy_instance(self.state.instance_id)
            self.state.instance_id = None
            self.state.status = "destroyed"
            self._save_state()
            return True

        except Exception as e:
            logger.error("destroy_failed", error=str(e))
            return False

    def run_loop(self, interval_seconds: int = 60) -> None:
        """Run the manager loop continuously.

        Checks instance status and provisions new ones as needed.

        Args:
            interval_seconds: Seconds between checks.
        """
        logger.info("manager_loop_started", interval=interval_seconds)

        try:
            while True:
                try:
                    self.ensure_worker_running()
                except Exception as e:
                    logger.error("loop_iteration_failed", error=str(e))

                logger.debug("sleeping", seconds=interval_seconds)
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            logger.info("manager_loop_stopped")

    def close(self) -> None:
        """Clean up resources."""
        self.client.close()
