"""Vast.ai API client using the official SDK.

Provides a wrapper around the vastai-sdk for:
- Listing available GPU offers
- Creating and managing instances
- Monitoring instance status
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field
from vastai import VastAI

from tsgb.logging import get_logger
from tsgb.settings import Settings

logger = get_logger(__name__)


class Offer(BaseModel):
    """A Vast.ai GPU offer."""

    id: int
    gpu_name: str
    num_gpus: int
    gpu_ram: float  # GB
    cpu_cores: int
    cpu_ram: float  # GB
    disk_space: float  # GB
    dph_total: float  # USD per hour
    dlperf: float | None = None  # Deep learning performance score
    dlperf_per_dphtotal: float | None = None  # DLPerf per dollar (from API)
    reliability: float | None = None
    inet_down: float | None = None  # Mbps
    inet_up: float | None = None  # Mbps
    cuda_max_good: float | str | None = None
    verified: bool = False
    static_ip: bool = False
    direct_port_count: int = 0

    @property
    def price_per_hour(self) -> float:
        """Get price per hour in USD."""
        return self.dph_total

    @property
    def dlperf_per_dollar(self) -> float:
        """Get DL performance per dollar (higher is better value).

        This is the key metric for finding cost-effective machines:
        dlperf / dph_total = performance per dollar per hour.
        """
        if self.dlperf is None or self.dph_total <= 0:
            return 0.0
        return self.dlperf / self.dph_total


class Instance(BaseModel):
    """A Vast.ai instance."""

    id: int
    machine_id: int | None = None
    status: str  # "running", "loading", "exited", etc.
    actual_status: str | None = None
    gpu_name: str | None = None
    num_gpus: int | None = None
    gpu_ram: float | None = None
    cpu_cores: int | None = None
    cpu_ram: float | None = None
    disk_space: float | None = None
    dph_total: float | None = None
    image_uuid: str | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    jupyter_token: str | None = None
    start_date: float | None = None
    end_date: float | None = None
    cur_state: str | None = None
    label: str | None = None
    extra_env: dict[str, str] = Field(default_factory=dict)

    @property
    def is_running(self) -> bool:
        """Check if instance is running."""
        return self.status == "running" or self.actual_status == "running"

    @property
    def is_dead(self) -> bool:
        """Check if instance is dead or exited."""
        return self.status in ("exited", "stopped", "destroyed", "error")


class VastAPIError(Exception):
    """Error from Vast.ai API."""

    def __init__(
        self, message: str, status_code: int | None = None, response: dict[str, Any] | None = None
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(message)


class VastAPIClient:
    """Client for Vast.ai API using the official SDK."""

    def __init__(
        self,
        api_key: str | None = None,
    ) -> None:
        """Initialize the Vast.ai API client.

        Args:
            api_key: Your Vast.ai API key. If not provided, uses VAST_API_KEY env var.
        """
        self.api_key = api_key
        # SDK reads from VAST_API_KEY env var if api_key is None
        if api_key:
            os.environ["VAST_API_KEY"] = api_key
        self._client = VastAI(api_key=api_key)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "VastAPIClient":
        """Create client from settings.

        Args:
            settings: Application settings. Uses defaults if not provided.

        Returns:
            Configured VastAPIClient.
        """
        settings = settings or Settings()
        return cls(api_key=settings.vast_api_key)

    def list_offers(
        self,
        gpu_name: str | None = None,
        min_vram_gb: int | None = None,
        max_price: float | None = None,
        num_gpus: int = 1,
        verified: bool | None = None,
        order_by: str = "dlperf_per_dphtotal",
        order_dir: str = "desc",
        limit: int = 20,
    ) -> list[Offer]:
        """List available GPU offers.

        Args:
            gpu_name: Filter by GPU name (e.g., "RTX_4090").
            min_vram_gb: Minimum VRAM in GB.
            max_price: Maximum price per hour in USD.
            num_gpus: Number of GPUs required.
            verified: Filter by verification status.
            order_by: Sort field (default: dlperf_per_dphtotal for best value).
            order_dir: Sort direction ("asc" or "desc", default: "desc" for best first).
            limit: Maximum number of offers to return.

        Returns:
            List of matching offers, sorted by specified field.
        """
        logger.info(
            "listing_offers",
            gpu_name=gpu_name,
            min_vram_gb=min_vram_gb,
            max_price=max_price,
            num_gpus=num_gpus,
            order_by=order_by,
            limit=limit,
        )

        try:
            # SDK search_offers with simple query, then filter in Python
            # The SDK query parser has issues with complex queries
            order_str = f"{order_by}-" if order_dir == "desc" else order_by

            # Fetch more than needed since we'll filter
            fetch_limit = limit * 5 if limit < 100 else 500

            results = self._client.search_offers(
                query="rentable = true",
                order=order_str,
                limit=fetch_limit,
                type="on-demand",
            )

            if not results:
                logger.info("no_offers_found")
                return []

            offers = []
            for offer_data in results:
                try:
                    gpu_ram_gb = offer_data.get("gpu_ram", 0) / 1024
                    dph = offer_data.get("dph_total", 0)
                    offer_num_gpus = offer_data.get("num_gpus", 1)
                    offer_gpu_name = offer_data.get("gpu_name", "unknown")
                    is_verified = offer_data.get("verified", False)

                    # Apply filters
                    if min_vram_gb and gpu_ram_gb < min_vram_gb:
                        continue
                    if max_price and dph > max_price:
                        continue
                    if num_gpus and offer_num_gpus < num_gpus:
                        continue
                    if gpu_name and offer_gpu_name != gpu_name:
                        continue
                    if verified is not None and is_verified != verified:
                        continue

                    offer = Offer(
                        id=offer_data.get("id", 0),
                        gpu_name=offer_gpu_name,
                        num_gpus=offer_num_gpus,
                        gpu_ram=gpu_ram_gb,
                        cpu_cores=int(offer_data.get("cpu_cores", 0)),
                        cpu_ram=offer_data.get("cpu_ram", 0) / 1024,
                        disk_space=offer_data.get("disk_space", 0),
                        dph_total=dph,
                        dlperf=offer_data.get("dlperf"),
                        dlperf_per_dphtotal=offer_data.get("dlperf_per_dphtotal"),
                        reliability=offer_data.get("reliability"),
                        inet_down=offer_data.get("inet_down"),
                        inet_up=offer_data.get("inet_up"),
                        cuda_max_good=offer_data.get("cuda_max_good"),
                        verified=is_verified,
                        static_ip=offer_data.get("static_ip", False),
                        direct_port_count=offer_data.get("direct_port_count", 0),
                    )
                    offers.append(offer)

                    # Stop if we have enough
                    if len(offers) >= limit:
                        break

                except Exception as e:
                    logger.warning(
                        "parse_offer_failed", error=str(e), offer_id=offer_data.get("id")
                    )

            logger.info("offers_found", count=len(offers))
            return offers

        except Exception as e:
            logger.error("list_offers_failed", error=str(e))
            raise VastAPIError(f"Failed to list offers: {e}") from e

    def create_instance(
        self,
        offer_id: int,
        image: str,
        disk_gb: int = 20,
        onstart: str | None = None,
        label: str | None = None,
        env: dict[str, str] | None = None,
    ) -> Instance:
        """Create a new instance from an offer.

        Args:
            offer_id: The offer ID to rent.
            image: Docker image to use.
            disk_gb: Disk space in GB.
            onstart: Startup script (bash).
            label: Instance label.
            env: Environment variables.

        Returns:
            The created instance.

        Raises:
            VastAPIError: On API errors.
        """
        logger.info(
            "creating_instance",
            offer_id=offer_id,
            image=image,
            disk_gb=disk_gb,
        )

        try:
            # Build environment string for SDK
            env_str = None
            if env:
                env_str = " ".join(f"-e {k}={v}" for k, v in env.items())

            # SDK create_instance
            result = self._client.create_instance(
                ID=offer_id,
                image=image,
                disk=disk_gb,
                onstart=onstart,
                label=label,
                env=env_str,
            )

            # Result should contain the new instance info
            if isinstance(result, dict):
                instance_id = result.get("new_contract") or result.get("id")
            else:
                # Result might be the instance ID directly
                instance_id = int(result) if result else None

            if not instance_id:
                raise VastAPIError("No instance ID returned from create request")

            logger.info("instance_created", instance_id=instance_id)

            # Fetch the instance details
            return self.get_instance(instance_id)

        except Exception as e:
            logger.error("create_instance_failed", error=str(e))
            raise VastAPIError(f"Failed to create instance: {e}") from e

    def get_instance(self, instance_id: int) -> Instance:
        """Get instance details.

        Args:
            instance_id: The instance ID.

        Returns:
            Instance details.

        Raises:
            VastAPIError: On API errors.
        """
        try:
            result = self._client.show_instance(id=instance_id)

            if not result:
                raise VastAPIError(f"Instance {instance_id} not found")

            instance_data = result if isinstance(result, dict) else {}

            return Instance(
                id=instance_data.get("id", instance_id),
                machine_id=instance_data.get("machine_id"),
                status=instance_data.get("status", "unknown"),
                actual_status=instance_data.get("actual_status"),
                gpu_name=instance_data.get("gpu_name"),
                num_gpus=instance_data.get("num_gpus"),
                gpu_ram=instance_data.get("gpu_ram", 0) / 1024
                if instance_data.get("gpu_ram")
                else None,
                cpu_cores=instance_data.get("cpu_cores"),
                cpu_ram=instance_data.get("cpu_ram", 0) / 1024
                if instance_data.get("cpu_ram")
                else None,
                disk_space=instance_data.get("disk_space"),
                dph_total=instance_data.get("dph_total"),
                image_uuid=instance_data.get("image_uuid"),
                ssh_host=instance_data.get("ssh_host"),
                ssh_port=instance_data.get("ssh_port"),
                jupyter_token=instance_data.get("jupyter_token"),
                start_date=instance_data.get("start_date"),
                end_date=instance_data.get("end_date"),
                cur_state=instance_data.get("cur_state"),
                label=instance_data.get("label"),
                extra_env=instance_data.get("extra_env", {}),
            )

        except VastAPIError:
            raise
        except Exception as e:
            logger.error("get_instance_failed", error=str(e))
            raise VastAPIError(f"Failed to get instance: {e}") from e

    def list_instances(self) -> list[Instance]:
        """List all user instances.

        Returns:
            List of instances.
        """
        try:
            results = self._client.show_instances()

            if not results:
                return []

            instances = []
            for instance_data in results:
                try:
                    instance = Instance(
                        id=instance_data.get("id", 0),
                        machine_id=instance_data.get("machine_id"),
                        status=instance_data.get("status", "unknown"),
                        actual_status=instance_data.get("actual_status"),
                        gpu_name=instance_data.get("gpu_name"),
                        num_gpus=instance_data.get("num_gpus"),
                        gpu_ram=instance_data.get("gpu_ram", 0) / 1024
                        if instance_data.get("gpu_ram")
                        else None,
                        cpu_cores=instance_data.get("cpu_cores"),
                        cpu_ram=instance_data.get("cpu_ram", 0) / 1024
                        if instance_data.get("cpu_ram")
                        else None,
                        disk_space=instance_data.get("disk_space"),
                        dph_total=instance_data.get("dph_total"),
                        image_uuid=instance_data.get("image_uuid"),
                        ssh_host=instance_data.get("ssh_host"),
                        ssh_port=instance_data.get("ssh_port"),
                        start_date=instance_data.get("start_date"),
                        end_date=instance_data.get("end_date"),
                        cur_state=instance_data.get("cur_state"),
                        label=instance_data.get("label"),
                    )
                    instances.append(instance)
                except Exception as e:
                    logger.warning("parse_instance_failed", error=str(e))

            return instances

        except Exception as e:
            logger.error("list_instances_failed", error=str(e))
            raise VastAPIError(f"Failed to list instances: {e}") from e

    def destroy_instance(self, instance_id: int) -> None:
        """Destroy an instance.

        Args:
            instance_id: The instance ID to destroy.

        Raises:
            VastAPIError: On API errors.
        """
        logger.info("destroying_instance", instance_id=instance_id)

        try:
            self._client.destroy_instance(id=instance_id)
            logger.info("instance_destroyed", instance_id=instance_id)
        except Exception as e:
            logger.error("destroy_instance_failed", error=str(e))
            raise VastAPIError(f"Failed to destroy instance: {e}") from e

    def stop_instance(self, instance_id: int) -> None:
        """Stop an instance (can be restarted).

        Args:
            instance_id: The instance ID to stop.

        Raises:
            VastAPIError: On API errors.
        """
        logger.info("stopping_instance", instance_id=instance_id)

        try:
            self._client.stop_instance(ID=instance_id)
            logger.info("instance_stopped", instance_id=instance_id)
        except Exception as e:
            logger.error("stop_instance_failed", error=str(e))
            raise VastAPIError(f"Failed to stop instance: {e}") from e

    def start_instance(self, instance_id: int) -> None:
        """Start a stopped instance.

        Args:
            instance_id: The instance ID to start.

        Raises:
            VastAPIError: On API errors.
        """
        logger.info("starting_instance", instance_id=instance_id)

        try:
            self._client.start_instance(ID=instance_id)
            logger.info("instance_started", instance_id=instance_id)
        except Exception as e:
            logger.error("start_instance_failed", error=str(e))
            raise VastAPIError(f"Failed to start instance: {e}") from e

    def get_instance_logs(self, instance_id: int, tail: int = 1000) -> str:
        """Get logs for an instance.

        Args:
            instance_id: The instance ID.
            tail: Number of lines to show from the end of the logs.

        Returns:
            Log output as a string.

        Raises:
            VastAPIError: On API errors.
        """
        logger.info("fetching_instance_logs", instance_id=instance_id, tail=tail)

        try:
            result = self._client.logs(INSTANCE_ID=instance_id, tail=str(tail))
            # SDK returns logs as string
            if isinstance(result, str):
                return result
            elif isinstance(result, dict):
                return result.get("result", "") or result.get("logs", "") or str(result)
            return str(result) if result else ""
        except Exception as e:
            logger.error("get_logs_failed", error=str(e))
            raise VastAPIError(f"Failed to get logs: {e}") from e

    def execute(self, instance_id: int, command: str) -> str:
        """Execute a command on an instance.

        Args:
            instance_id: The instance ID.
            command: Command to execute.

        Returns:
            Command output.

        Raises:
            VastAPIError: On API errors.
        """
        logger.info("executing_command", instance_id=instance_id, command=command[:50])

        try:
            result = self._client.execute(ID=instance_id, COMMAND=command)
            if isinstance(result, str):
                return result
            return str(result) if result else ""
        except Exception as e:
            logger.error("execute_failed", error=str(e))
            raise VastAPIError(f"Failed to execute command: {e}") from e

    def close(self) -> None:
        """Close the client (no-op for SDK, kept for compatibility)."""
        pass

    def __enter__(self) -> "VastAPIClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
