from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Any

from questionary import Choice
from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from modal_newbie_train import (
    DEFAULT_HF_REPO,
    DEFAULT_HF_SECRET,
    DEFAULT_VOLUME,
    TrainJob,
    download_hf_model_to_volume,
    download_job_output,
    run_remote_training,
)
from scripts.billing import format_money
from scripts.config_flow import ask_dataset_directory, ask_sanitized_name, choose_config
from scripts.preferences import load_preferences, save_preferences
from scripts.tui import (
    ask_confirm,
    ask_positive_int,
    ask_select,
    ask_text,
    console,
    print_log_tail,
    print_result_panel,
    print_status,
    print_step,
    validate_required,
)


GPU_SECOND_RATES = {
    "H100": Decimal("0.001097"),
    "A100-80GB": Decimal("0.000694"),
    "A100-40GB": Decimal("0.000583"),
    "L40S": Decimal("0.000542"),
    "T4": Decimal("0.000164"),
}


def status_label(ok: Any) -> str:
    if ok is True:
        return "[bold green]SUCCESS[/bold green]"
    if ok is False:
        return "[bold red]FAILED[/bold red]"
    return str(ok)


def dashboard_value(value: Any) -> str | None:
    if not value:
        return None
    return f"[bold cyan]{value}[/bold cyan]"


def gpu_label(gpu: str) -> str:
    if "100" in gpu:
        return f"[bold reverse red] {gpu} [/bold reverse red]"
    return f"[bold cyan]{gpu}[/bold cyan]"


def estimated_max_gpu_cost(gpu: str, timeout_minutes: int) -> str:
    rate = GPU_SECOND_RATES.get(gpu)
    if rate is None:
        return "[dim]Unavailable[/dim]"
    cost = rate * Decimal(timeout_minutes * 60)
    return f"{format_money(cost)} [dim]GPU max by timeout[/dim]"


def dataset_review_value(job: TrainJob) -> str:
    if job.upload and job.dataset_path:
        return f"[bold green]UPLOAD NEW[/bold green] [dim]{job.dataset_path}[/dim]"
    return "[dim]REUSE VOLUME[/dim] Reusing dataset already in Modal Volume"


def launch_mode_label(detach: bool) -> str:
    if detach:
        return "[magenta]Detached (Background)[/magenta]"
    return "[white]Attached (Live Logs)[/white]"


def print_training_review(job: TrainJob) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim", no_wrap=True)
    table.add_column("Value", style="bold white", overflow="fold")
    table.add_row("Job Name", f"[bold white]{job.name}[/bold white]")
    table.add_row("Config", str(job.config_path))
    table.add_row("Dataset", dataset_review_value(job))
    table.add_row("Upload", "[bold green]Yes[/bold green]" if job.upload else "[dim]No[/dim]")
    table.add_row("GPU Type", gpu_label(job.gpu))
    table.add_row("Timeout", f"{job.timeout_minutes} mins")
    table.add_row("Estimated Max GPU Cost", estimated_max_gpu_cost(job.gpu, job.timeout_minutes))
    table.add_row("Launch Mode", launch_mode_label(job.detach))
    table.add_row("Dependencies", "[dim]Baked image[/dim]")
    note = (
        "[dim]Cost estimate uses Modal public GPU pricing and the timeout limit; "
        "actual billing may include CPU, memory, storage, credits, or discounts.[/dim]"
    )
    console.print(Panel(Group(table, note), title="[bold yellow]PRE-FLIGHT CHECK[/bold yellow]", border_style="yellow"))


def run_training_flow() -> None:
    # This flow collects local choices and delegates all Modal work to the headless runner.
    preferences = load_preferences()
    print_step("Step 1: Job Identity")
    config = choose_config(preferences.get("last_config"))
    job_name = ask_sanitized_name("Modal job name", config.stem, "job name")

    print_step("Step 2: Inputs")
    dataset_source = ask_select(
        "Dataset source",
        [
            Choice("Upload local dataset", value="upload", description="Upload a local dataset folder for this job."),
            Choice("Reuse dataset already in Volume", value="reuse", description="Use the existing /jobs/<job>/dataset in Modal Volume."),
        ],
    )
    upload = dataset_source == "upload"
    dataset_path = ask_dataset_directory() if upload else None

    print_step("Step 3: Resources")
    preferred_gpu = str(preferences.get("last_gpu") or "L40S")
    gpu = ask_select(
        "GPU",
        [
            Choice("L40S", value="L40S", description="Recommended default for cost and speed."),
            Choice("A100-40GB", value="A100-40GB", description="More memory for heavier runs."),
            Choice("A100-80GB", value="A100-80GB", description="Large-memory A100 option."),
            Choice("H100", value="H100", description="Fastest option with higher cost."),
            Choice("T4", value="T4", description="Lower-cost option for small tests."),
        ],
        default=preferred_gpu if preferred_gpu in GPU_SECOND_RATES else "L40S",
    )
    timeout_default = str(preferences.get("last_timeout") or "360")
    timeout = ask_positive_int("Timeout minutes", timeout_default, "Timeout minutes")

    print_step("Step 4: Launch Options")
    preferred_run_mode = str(preferences.get("last_run_mode") or "attached")
    run_mode = ask_select(
        "Launch mode",
        [
            Choice("Attached (Live Logs)", value="attached", description="Stream Modal logs in this terminal until training finishes."),
            Choice("Detached (Background)", value="detached", description="Submit the job and let Modal continue in the background."),
        ],
        default=preferred_run_mode if preferred_run_mode in {"attached", "detached"} else "attached",
    )
    detach = run_mode == "detached"

    job = TrainJob(
        name=job_name,
        config_path=config,
        dataset_path=dataset_path,
        gpu=gpu,
        timeout_minutes=timeout,
        install_requirements=False,
        upload=upload,
        detach=detach,
    )

    print_step("Step 5: Summary")
    print_training_review(job)
    if not ask_confirm(
        "Is this configuration correct?",
        True,
        instruction="Choose No to cancel and return to the main menu.",
    ):
        print_status(
            "[yellow]Training submission canceled by user. You can re-enter 'Run training' to fix settings.[/yellow]",
            style="yellow",
        )
        return

    save_preferences(
        {
            "last_config": str(config),
            "last_gpu": gpu,
            "last_timeout": timeout,
            "last_run_mode": run_mode,
        }
    )
    if detach:
        with console.status("[bold blue]Submitting training job to Modal...[/bold blue]", spinner="dots"):
            result = run_remote_training(job)
    else:
        print_status("[bold blue]Starting remote training. Live logs may stream below.[/bold blue]", style="blue")
        result = run_remote_training(job)
    if result.get("submitted"):
        print_result_panel(
            "[bold green]Training Job Submitted[/bold green]",
            [
                ("Status", "[bold green]SUBMITTED[/bold green]"),
                ("Detached", result.get("detached")),
                ("App ID", result.get("app_id")),
                ("Function Call ID", result.get("function_call_id")),
                ("Dashboard", dashboard_value(result.get("app_dashboard_url"))),
                ("Function Call", dashboard_value(result.get("function_call_dashboard_url"))),
                ("Output", result.get("output_dir")),
                ("Log", result.get("log_path")),
            ],
        )
        return

    ok = result.get("ok")
    print_result_panel(
        "[bold green]Remote Training Finished[/bold green]" if ok else "[bold red]Remote Training Failed[/bold red]",
        [
            ("Status", status_label(ok)),
            ("Output", result.get("output_dir")),
            ("Log", result.get("log_path")),
            ("Local App Log", result.get("local_app_log_path")),
            ("Local Zip", result.get("local_zip")),
        ],
        border_style="green" if ok else "red",
    )
    if not result.get("ok"):
        print_log_tail(result.get("log_tail", ""))


def load_model_flow() -> None:
    # Newbie training configs expect the base model to live at /workspace/Models.
    console.print(
        Panel(
            f"[dim]Model:[/dim] [bold cyan]{DEFAULT_HF_REPO}[/bold cyan]\n"
            "[dim]Target:[/dim] [bold cyan]/workspace/Models[/bold cyan]",
            title="[bold cyan]Base Model[/bold cyan]",
            border_style="cyan",
        )
    )
    timeout = ask_positive_int("Timeout minutes", "360", "Timeout minutes")
    default_secret = os.environ.get("MODAL_HF_SECRET_NAME", "").strip() or DEFAULT_HF_SECRET
    hf_secret = ask_text(
        "Modal Secret name for HF_TOKEN",
        default_secret,
        validate=validate_required("Modal Secret name"),
    )
    result = download_hf_model_to_volume(DEFAULT_HF_REPO, None, DEFAULT_VOLUME, timeout, hf_secret)

    ok = result.get("ok")
    print_result_panel(
        "[bold green]Model Load Finished[/bold green]" if ok else "[bold red]Model Load Failed[/bold red]",
        [
            ("Status", status_label(ok)),
            ("Remote Path", result.get("remote_path")),
            ("Files", result.get("file_count")),
            ("Bytes", result.get("bytes")),
            ("Error", result.get("error")),
        ],
        border_style="green" if ok else "red",
    )


def download_job_output_flow() -> None:
    # Download the final adapter folder inferred from the job name and config.
    config = choose_config()
    job_name = ask_sanitized_name("Modal job name used for training", config.stem, "job name")
    output_name = ask_text(
        "Remote output folder override",
        "",
        instruction="Leave blank to use Model.output_name from the selected config.",
    )
    local_path = ask_text(
        "Local destination",
        "",
        instruction="Leave blank to use outputs/<job>/<output>.",
    )

    try:
        result = download_job_output(
            DEFAULT_VOLUME,
            job_name,
            config,
            Path(local_path) if local_path else None,
            output_name or None,
        )
    except FileNotFoundError as exc:
        console.print(
            Panel(
                f"{exc}\n\n[dim]Leave the override blank to use Model.output_name from the selected config.[/dim]",
                title="[bold red]Download Failed[/bold red]",
                border_style="red",
            )
        )
        return

    print_result_panel(
        "[bold green]Job Output Downloaded[/bold green]",
        [
            ("Remote Path", result["remote_path"]),
            ("Local Path", result["local_path"]),
            ("Bytes", result["bytes"]),
        ],
    )
