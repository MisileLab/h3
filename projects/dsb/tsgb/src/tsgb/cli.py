"""Command-line interface for TSGB.

Provides CLI commands for:
- Local training (train-local)
- Local evaluation (eval-local)
- Vast.ai manager (manager run/loop)
- Worker process (worker run)
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from tsgb.logging import configure_logging, get_logger
from tsgb.settings import get_settings

# Create Typer app
app = typer.Typer(
    name="tsgb",
    help="Train Small, Guard Big - LLM jailbreak attack/defense framework",
    no_args_is_help=True,
)

# Sub-apps for manager and worker
manager_app = typer.Typer(help="Vast.ai instance manager commands")
worker_app = typer.Typer(help="Training worker commands")

app.add_typer(manager_app, name="manager")
app.add_typer(worker_app, name="worker")

console = Console()


def setup_logging() -> None:
    """Set up logging based on settings."""
    settings = get_settings()
    configure_logging(mode=settings.log_mode)  # type: ignore


# =============================================================================
# Local Training Command
# =============================================================================


@app.command("train-local")
def train_local(
    episodes: int = typer.Option(100, "--episodes", "-e", help="Number of training episodes"),
    checkpoint_interval: int = typer.Option(
        10, "--checkpoint-interval", "-c", help="Save checkpoint every N episodes"
    ),
    log_interval: int = typer.Option(
        5, "--log-interval", "-l", help="Log metrics every N episodes"
    ),
    checkpoint_dir: str = typer.Option(
        "./checkpoints", "--checkpoint-dir", "-d", help="Checkpoint directory"
    ),
    resume: str | None = typer.Option(None, "--resume", "-r", help="Resume from checkpoint path"),
    model_name: str | None = typer.Option(
        None, "--model", "-m", help="Model name (default: from settings)"
    ),
) -> None:
    """Run Stage 1 self-play RL training locally."""
    setup_logging()
    logger = get_logger(__name__)

    console.print("[bold blue]TSGB Local Training[/bold blue]")
    console.print(f"Episodes: {episodes}")
    console.print(f"Checkpoint interval: {checkpoint_interval}")
    console.print(f"Checkpoint directory: {checkpoint_dir}")

    if resume:
        console.print(f"Resuming from: {resume}")

    # Import here to avoid circular imports and speed up CLI startup
    from tsgb.trainer import TrainConfig, SelfPlayTrainer
    from tsgb.models import HuggingFaceLM, LLMRole

    settings = get_settings()

    config = TrainConfig(
        total_episodes=episodes,
        checkpoint_interval=checkpoint_interval,
        log_interval=log_interval,
        checkpoint_dir=checkpoint_dir,
        batch_size=settings.batch_size,
    )
    model = model_name or settings.default_model_name

    console.print(f"[yellow]Loading model: {model}[/yellow]")

    attacker = HuggingFaceLM.from_pretrained(model, role=LLMRole.ATTACKER)
    guard = HuggingFaceLM.from_pretrained(model, role=LLMRole.GUARD)
    target = HuggingFaceLM.from_pretrained(model, role=LLMRole.TARGET)

    trainer = SelfPlayTrainer(
        attacker_model=attacker,
        guard_model=guard,
        target_model=target,
        config=config,
    )

    try:
        console.print("\n[bold]Starting training...[/bold]")
        trainer.train(resume_from=resume)
        console.print("\n[bold green]Training completed![/bold green]")
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user.[/yellow]")


# =============================================================================
# Local Evaluation Command
# =============================================================================


@app.command("eval-local")
def eval_local(
    checkpoint_path: str | None = typer.Option(
        None, "--checkpoint-path", "-c", help="Path to guard checkpoint"
    ),
    num_benign: int = typer.Option(50, "--num-benign", help="Number of benign test samples"),
    num_attacks: int = typer.Option(50, "--num-attacks", help="Number of attack test samples"),
    dataset_path: str | None = typer.Option(
        None, "--dataset", "-d", help="Path to evaluation dataset (JSON or HuggingFace ID)"
    ),
    blackbox_provider: str = typer.Option(
        "openai", "--provider", "-p", help="Black-box LLM provider (openai, anthropic, google)"
    ),
    blackbox_model: str | None = typer.Option(
        None, "--blackbox-model", help="Black-box model name (provider-specific)"
    ),
) -> None:
    """Run Stage 2 evaluation with black-box LLM."""
    setup_logging()

    console.print("[bold blue]TSGB Stage 2 Evaluation[/bold blue]")
    console.print(f"Benign samples: {num_benign}")
    console.print(f"Attack samples: {num_attacks}")
    console.print(f"Black-box provider: {blackbox_provider}")

    if checkpoint_path:
        console.print(f"Checkpoint: {checkpoint_path}")

    # Import here for faster CLI startup
    from tsgb.eval import (
        OpenAILLM,
        AnthropicLLM,
        GoogleLLM,
        load_evaluation_dataset,
        evaluate_guard_on_blackbox,
    )
    from tsgb.models import HuggingFaceLM, LLMRole

    settings = get_settings()

    # Load guard model
    if checkpoint_path:
        console.print(f"[yellow]Loading guard from checkpoint: {checkpoint_path}[/yellow]")
        # Load from checkpoint - for now use default model as base
        guard = HuggingFaceLM.from_pretrained(
            settings.default_model_name,
            role=LLMRole.GUARD,
        )
        # TODO: Load weights from checkpoint
    else:
        console.print(f"[yellow]Loading fresh guard model: {settings.default_model_name}[/yellow]")
        guard = HuggingFaceLM.from_pretrained(
            settings.default_model_name,
            role=LLMRole.GUARD,
        )

    # Create black-box LLM
    console.print(f"[yellow]Initializing {blackbox_provider} black-box LLM...[/yellow]")
    if blackbox_provider == "openai":
        blackbox = OpenAILLM(model=blackbox_model) if blackbox_model else OpenAILLM()
    elif blackbox_provider == "anthropic":
        blackbox = AnthropicLLM(model=blackbox_model) if blackbox_model else AnthropicLLM()
    elif blackbox_provider == "google":
        blackbox = GoogleLLM(model=blackbox_model) if blackbox_model else GoogleLLM()
    else:
        console.print(f"[red]Unknown provider: {blackbox_provider}[/red]")
        raise typer.Exit(1)

    # Load dataset
    dataset = load_evaluation_dataset(
        dataset_path=dataset_path,
        num_benign=num_benign,
        num_attacks=num_attacks,
    )

    console.print(f"\nTotal samples: {len(dataset)}")
    console.print("\n[bold]Running evaluation...[/bold]")

    # Run evaluation
    metrics = evaluate_guard_on_blackbox(
        guard=guard,
        blackbox=blackbox,
        dataset=dataset,
    )

    # Display results
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    results = metrics.to_dict()
    for key, value in results.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))

    console.print(table)


# =============================================================================
# Manager Commands
# =============================================================================


@manager_app.command("run")
def manager_run(
    repo_url: str = typer.Option(
        "https://gith.misile.xyz/h3.git:/projects/dsb/tsgb.git",
        "--repo-url",
        help="Git repository URL",
    ),
    branch: str = typer.Option("main", "--branch", help="Git branch"),
    state_file: str = typer.Option("manager_state.json", "--state-file", help="Manager state file"),
) -> None:
    """Check instance status and provision if needed (single run)."""
    setup_logging()

    console.print("[bold blue]TSGB Instance Manager[/bold blue]")

    from tsgb.manager import InstanceManager

    settings = get_settings()
    manager = InstanceManager(settings=settings, state_file=state_file)

    try:
        instance = manager.ensure_worker_running(repo_url=repo_url, branch=branch)

        if instance:
            table = Table(title="Instance Status")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Instance ID", str(instance.id))
            table.add_row("Status", instance.status)
            table.add_row("GPU", instance.gpu_name or "N/A")
            table.add_row("GPU RAM", f"{instance.gpu_ram:.1f} GB" if instance.gpu_ram else "N/A")
            table.add_row("Price", f"${instance.dph_total:.3f}/hr" if instance.dph_total else "N/A")

            console.print(table)
        else:
            console.print("[yellow]No instance available.[/yellow]")

    finally:
        manager.close()


@manager_app.command("loop")
def manager_loop(
    interval: int = typer.Option(60, "--interval", "-i", help="Check interval in seconds"),
    repo_url: str = typer.Option(
        "https://gith.misile.xyz/h3.git:/projects/dsb/tsgb.git",
        "--repo-url",
        help="Git repository URL",
    ),
    branch: str = typer.Option("main", "--branch", help="Git branch"),
    state_file: str = typer.Option("manager_state.json", "--state-file", help="Manager state file"),
) -> None:
    """Run manager loop continuously."""
    setup_logging()

    console.print("[bold blue]TSGB Instance Manager Loop[/bold blue]")
    console.print(f"Check interval: {interval}s")
    console.print("Press Ctrl+C to stop.\n")

    from tsgb.manager import InstanceManager

    settings = get_settings()
    manager = InstanceManager(settings=settings, state_file=state_file)

    try:
        manager.run_loop(interval_seconds=interval)
    except KeyboardInterrupt:
        console.print("\n[yellow]Manager stopped.[/yellow]")
    finally:
        manager.close()


@manager_app.command("status")
def manager_status(
    state_file: str = typer.Option("manager_state.json", "--state-file", help="Manager state file"),
) -> None:
    """Show current manager and instance status."""
    setup_logging()

    from tsgb.manager import InstanceManager

    settings = get_settings()
    manager = InstanceManager(settings=settings, state_file=state_file)

    try:
        console.print("[bold blue]Manager Status[/bold blue]\n")

        # Manager state
        state_table = Table(title="Manager State")
        state_table.add_column("Field", style="cyan")
        state_table.add_column("Value", style="green")

        state_table.add_row("State File", str(manager.state_file))
        state_table.add_row("Instance ID", str(manager.state.instance_id) or "None")
        state_table.add_row("Status", manager.state.status)
        state_table.add_row("Last Check", manager.state.last_check_at or "Never")
        state_table.add_row("Last Provision", manager.state.last_provision_at or "Never")

        console.print(state_table)

        # Check actual instance status
        if manager.state.instance_id:
            console.print("\n[bold]Fetching instance details...[/bold]")
            instance = manager.check_instance_status()

            if instance:
                instance_table = Table(title="Instance Details")
                instance_table.add_column("Field", style="cyan")
                instance_table.add_column("Value", style="green")

                instance_table.add_row("ID", str(instance.id))
                instance_table.add_row("Status", instance.status)
                instance_table.add_row("Actual Status", instance.actual_status or "N/A")
                instance_table.add_row("GPU", instance.gpu_name or "N/A")
                instance_table.add_row("SSH Host", instance.ssh_host or "N/A")
                instance_table.add_row(
                    "SSH Port", str(instance.ssh_port) if instance.ssh_port else "N/A"
                )

                console.print(instance_table)

    finally:
        manager.close()


@manager_app.command("destroy")
def manager_destroy(
    state_file: str = typer.Option("manager_state.json", "--state-file", help="Manager state file"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Destroy the current worker instance."""
    setup_logging()

    from tsgb.manager import InstanceManager

    settings = get_settings()
    manager = InstanceManager(settings=settings, state_file=state_file)

    try:
        if manager.state.instance_id is None:
            console.print("[yellow]No instance to destroy.[/yellow]")
            return

        if not force:
            confirm = typer.confirm(
                f"Destroy instance {manager.state.instance_id}?",
                default=False,
            )
            if not confirm:
                console.print("[yellow]Cancelled.[/yellow]")
                return

        if manager.destroy_worker():
            console.print("[green]Instance destroyed.[/green]")
        else:
            console.print("[red]Failed to destroy instance.[/red]")

    finally:
        manager.close()


@manager_app.command("logs")
def manager_logs(
    instance_id: int | None = typer.Argument(
        None, help="Instance ID (uses tracked instance if not specified)"
    ),
    tail: int = typer.Option(1000, "--tail", "-n", help="Number of lines to show"),
    state_file: str = typer.Option("manager_state.json", "--state-file", help="Manager state file"),
) -> None:
    """Get logs for an instance."""
    setup_logging()

    from tsgb.vast_api import VastAPIClient

    settings = get_settings()

    # If no instance ID provided, try to get from state
    if instance_id is None:
        from tsgb.manager import InstanceManager

        manager = InstanceManager(settings=settings, state_file=state_file)
        instance_id = manager.state.instance_id
        manager.close()

        if instance_id is None:
            console.print("[red]No instance ID provided and no tracked instance.[/red]")
            raise typer.Exit(1)

    console.print(f"[bold blue]Fetching logs for instance {instance_id}...[/bold blue]\n")

    client = VastAPIClient.from_settings(settings)
    try:
        logs = client.get_instance_logs(instance_id, tail=tail)
        console.print(logs)
    except Exception as e:
        console.print(f"[red]Error fetching logs: {e}[/red]")
        raise typer.Exit(1)
    finally:
        client.close()


@manager_app.command("offers")
def manager_offers(
    limit: int = typer.Option(10, "--limit", "-l", help="Number of offers to show"),
    min_vram: int | None = typer.Option(None, "--min-vram", help="Minimum VRAM in GB"),
    max_price: float | None = typer.Option(None, "--max-price", help="Maximum $/hour"),
    instance_type: str = typer.Option(
        None,
        "--type",
        "-t",
        help="Instance type (on-demand, bid, reserved)",
    ),
    show_min_bid: bool = typer.Option(False, "--show-min-bid", help="Show min bid price"),
) -> None:
    """List available GPU offers sorted by DLPerf/$."""
    setup_logging()

    from tsgb.vast_api import VastAPIClient

    settings = get_settings()
    client = VastAPIClient.from_settings(settings)

    try:
        offers = client.list_offers(
            min_vram_gb=min_vram or settings.vast_min_vram_gb,
            max_price=max_price or settings.vast_max_price,
            verified=True,
            instance_type=instance_type or settings.vast_instance_type,
            order_by="dlperf_per_dphtotal",
            order_dir="desc",
            limit=limit,
        )

        if not offers:
            console.print("[yellow]No offers found matching criteria.[/yellow]")
            return

        table = Table(title="Available GPU Offers (sorted by DLPerf/$)")
        table.add_column("ID", style="cyan")
        table.add_column("GPU", style="green")
        table.add_column("GPUs", justify="right")
        table.add_column("VRAM (total)", justify="right")
        table.add_column("$/hr", justify="right")
        if show_min_bid:
            table.add_column("Min Bid", justify="right")
        table.add_column("DLPerf", justify="right")
        table.add_column("DLPerf/$", justify="right", style="bold yellow")
        table.add_column("Reliability", justify="right")

        for offer in offers:
            dlperf_str = f"{offer.dlperf:.1f}" if offer.dlperf else "N/A"
            dlperf_per_dollar = offer.dlperf_per_dphtotal or offer.dlperf_per_dollar
            dlperf_dollar_str = f"{dlperf_per_dollar:.1f}" if dlperf_per_dollar else "N/A"
            reliability_str = f"{offer.reliability:.2%}" if offer.reliability else "N/A"
            min_bid_str = f"${offer.min_bid:.3f}" if offer.min_bid is not None else "N/A"

            row = [
                str(offer.id),
                offer.gpu_name,
                str(offer.num_gpus),
                f"{offer.total_vram_gb:.0f}GB",
                f"${offer.dph_total:.3f}",
            ]
            if show_min_bid:
                row.append(min_bid_str)
            row.extend(
                [
                    dlperf_str,
                    dlperf_dollar_str,
                    reliability_str,
                ]
            )

            table.add_row(*row)

        console.print(table)

    finally:
        client.close()


# =============================================================================
# Worker Commands
# =============================================================================


@worker_app.command("run")
def worker_run(
    resume_path: str | None = typer.Option(
        None, "--resume-path", "-r", help="Checkpoint directory path"
    ),
    local_fallback: str | None = typer.Option(
        None, "--local-fallback", help="Fallback checkpoint path"
    ),
    episodes: int = typer.Option(1000, "--episodes", "-e", help="Total training episodes"),
    checkpoint_interval: int = typer.Option(
        100, "--checkpoint-interval", help="Save every N episodes"
    ),
    model_name: str | None = typer.Option(
        None, "--model", "-m", help="Model name (default: from settings)"
    ),
    trackio: bool | None = typer.Option(
        None,
        "--trackio/--no-trackio",
        help="Enable Trackio logging (defaults to settings)",
    ),
    trackio_project: str | None = typer.Option(
        None, "--trackio-project", help="Trackio project/run name"
    ),
    trackio_space_id: str | None = typer.Option(
        None, "--trackio-space-id", help="Trackio space ID"
    ),
) -> None:
    """Run training worker (typically on Vast.ai instance)."""
    setup_logging()

    console.print("[bold blue]TSGB Training Worker[/bold blue]")

    from tsgb.worker import run_worker

    run_worker(
        resume_path=resume_path,
        local_fallback=local_fallback,
        model_name=model_name,
        total_episodes=episodes,
        checkpoint_interval=checkpoint_interval,
        enable_trackio=trackio,
        trackio_project=trackio_project,
        trackio_space_id=trackio_space_id,
    )


# =============================================================================
# Utility Commands
# =============================================================================


@app.command("version")
def version() -> None:
    """Show version information."""
    from tsgb import __version__

    console.print(f"TSGB version {__version__}")


@app.command("config")
def show_config() -> None:
    """Show current configuration."""
    settings = get_settings()

    table = Table(title="Current Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # Show non-sensitive settings
    table.add_row("Vast.ai API Key", "***" if settings.vast_api_key else "(not set)")
    table.add_row("GPU Name", settings.vast_gpu_name or "(any)")
    table.add_row("Instance Type", settings.vast_instance_type)
    table.add_row("Min VRAM (GB)", str(settings.vast_min_vram_gb))
    table.add_row("Max Price ($/hr)", str(settings.vast_max_price))
    table.add_row("Checkpoint Dir", settings.checkpoint_dir)
    table.add_row("Local Checkpoint Dir", settings.local_checkpoint_dir)
    table.add_row("WebDAV URL", settings.rclone_webdav_url or "(not set)")
    table.add_row("WebDAV User", settings.rclone_webdav_user or "(not set)")
    table.add_row("Log Mode", settings.log_mode)
    table.add_row("Default Model", settings.default_model_name)

    console.print(table)


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
