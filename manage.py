from __future__ import annotations

from collections.abc import Callable, Sequence
import datetime as dt
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from pathlib import PurePosixPath
import subprocess
from typing import Any

import questionary
from questionary import Choice, Style
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

from modal_newbie_train import (
    DEFAULT_HF_REPO,
    DEFAULT_HF_SECRET,
    DEFAULT_VOLUME,
    TrainJob,
    delete_volume,
    download_job_output,
    download_hf_model_to_volume,
    get_volume_dashboard_url,
    list_all_volumes,
    list_volume,
    remove_job_directory,
    rename_volume,
    run_remote_training,
    safe_slug,
)


# Keep all TUI prompts on the same questionary theme so the menu reads as one tool.
STYLE = Style(
    [
        ("qmark", "fg:#58a6ff bold"),
        ("question", "fg:#f0f6fc bold"),
        ("answer", "fg:#7ee787 bold"),
        ("pointer", "fg:#f778ba bold"),
        ("highlighted", "fg:#f778ba bold"),
        ("selected", "fg:#7ee787"),
        ("separator", "fg:#30363d"),
        ("instruction", "fg:#8b949e italic"),
        ("text", "fg:#e6edf3"),
    ]
)

console = Console()

CONFIG_DIR = Path("configs")
JOB_CONFIG_DIR = CONFIG_DIR / "jobs"
GPU_SECOND_RATES = {
    "H100": Decimal("0.001097"),
    "A100-80GB": Decimal("0.000694"),
    "A100-40GB": Decimal("0.000583"),
    "L40S": Decimal("0.000542"),
    "T4": Decimal("0.000164"),
}

Validator = Callable[[str], bool | str]


def print_banner() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]Newbie-image[/bold cyan] [magenta]LoRA Modal Manager[/magenta]\n"
            "[dim]Terminal UI for Modal Training Workflows[/dim]",
            border_style="blue",
            padding=(1, 4),
        )
    )


def print_status(message: str, *, style: str = "green") -> None:
    console.print(Panel.fit(message, border_style=style, padding=(0, 2)))


def print_step(title: str) -> None:
    console.rule(f"[dim]{title}[/dim]", style="dim #30363d")


def print_result_panel(
    title: str,
    rows: Sequence[tuple[str, Any]],
    *,
    border_style: str = "green",
) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim", no_wrap=True)
    table.add_column("Value", style="bold white", overflow="fold")
    for key, value in rows:
        if value is None or value == "":
            continue
        table.add_row(key, str(value))
    console.print(Panel(table, title=title, border_style=border_style, box=box.ROUNDED))


def print_log_tail(log_tail: str) -> None:
    if log_tail:
        console.print(Panel(log_tail.rstrip(), title="Log Tail", border_style="yellow"))


def current_modal_profile() -> str | None:
    try:
        result = subprocess.run(
            ["modal", "profile", "current"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    profile = result.stdout.strip()
    return profile or None


def floor_hour(value: dt.datetime) -> dt.datetime:
    return value.replace(minute=0, second=0, microsecond=0)


def ceil_hour(value: dt.datetime) -> dt.datetime:
    return floor_hour(value) + dt.timedelta(hours=1)


def modal_datetime_arg(value: dt.datetime) -> str:
    return value.isoformat(timespec="seconds")


def fetch_billing_report(start: dt.datetime, end: dt.datetime) -> list[dict[str, Any]] | None:
    try:
        result = subprocess.run(
            [
                "modal",
                "billing",
                "report",
                "--start",
                modal_datetime_arg(start),
                "--end",
                modal_datetime_arg(end),
                "--resolution",
                "h",
                "--tz",
                "local",
                "--json",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return [row for row in parsed if isinstance(row, dict)]


def parse_cost(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def format_money(value: Decimal) -> str:
    return f"${value.quantize(Decimal('0.000001'))}"


def format_dt(value: dt.datetime) -> str:
    return value.isoformat(timespec="seconds")


def billing_summary_table(rows: list[dict[str, Any]]) -> tuple[Table, Decimal]:
    costs_by_description: dict[str, Decimal] = {}
    for row in rows:
        description = str(row.get("Description") or "Unlabeled")
        costs_by_description[description] = costs_by_description.get(description, Decimal("0")) + parse_cost(row.get("Cost"))

    total = sum(costs_by_description.values(), Decimal("0"))
    table = Table(show_header=True, box=box.SIMPLE_HEAVY, padding=(0, 2))
    table.add_column("Description", style="bold white", overflow="fold")
    table.add_column("Cost", justify="right", style="bold green", no_wrap=True)
    for description, cost in sorted(costs_by_description.items(), key=lambda item: item[1], reverse=True)[:5]:
        table.add_row(description, format_money(cost))
    return table, total


def print_exit_summary(session_start: dt.datetime, session_end: dt.datetime) -> None:
    profile = current_modal_profile()
    billing_start = floor_hour(session_start)
    billing_end = ceil_hour(session_end)
    rows = fetch_billing_report(billing_start, billing_end)

    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_column("Key", style="dim", no_wrap=True)
    summary.add_column("Value", style="bold white", overflow="fold")
    summary.add_row("Modal Profile", profile or "Unavailable")
    summary.add_row("Session Start", format_dt(session_start))
    summary.add_row("Session End", format_dt(session_end))
    summary.add_row("Billing Window", f"{format_dt(billing_start)} -> {format_dt(billing_end)}")
    summary.add_row("Dashboard", "https://modal.com/apps")

    if rows is None:
        summary.add_row("Billing", "[yellow]Unavailable[/yellow]")
        note = "[dim]Billing report failed or timed out; session exit was not blocked.[/dim]"
        console.print(Panel(Group(summary, note), title="[bold blue]Session Closed[/bold blue]", border_style="blue"))
        return

    billing_table, total = billing_summary_table(rows)
    summary.add_row("Total Reported Cost", f"[bold green]{format_money(total)}[/bold green]")
    summary.add_row("Rows", str(len(rows)))
    note = (
        "[dim]Modal reports full billing intervals only; the latest partial hour may appear later.[/dim]"
    )
    content = Group(
        summary,
        billing_table if rows else "[dim]No finalized billing rows for this session window yet.[/dim]",
        note,
    )
    console.print(
        Panel(
            content,
            title="[bold blue]Session Closed[/bold blue]",
            border_style="blue",
        )
    )


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


def dataset_review_value(job: TrainJob, *, first_upload: bool) -> str:
    if first_upload and job.dataset_path:
        return f"[bold green]UPLOAD NEW[/bold green] [dim]{job.dataset_path}[/dim]"
    return "[dim]REUSE VOLUME[/dim] Reusing dataset already in Modal Volume"


def print_training_review(job: TrainJob, *, first_upload: bool) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim", no_wrap=True)
    table.add_column("Value", style="bold white", overflow="fold")
    table.add_row("Job Name", f"[bold white]{job.name}[/bold white]")
    table.add_row("Config", str(job.config_path))
    table.add_row("Dataset", dataset_review_value(job, first_upload=first_upload))
    table.add_row("Upload", "[bold green]Yes[/bold green]" if job.upload else "[dim]No[/dim]")
    table.add_row("GPU Type", gpu_label(job.gpu))
    table.add_row("Timeout", f"{job.timeout_minutes} mins")
    table.add_row("Estimated Max GPU Cost", estimated_max_gpu_cost(job.gpu, job.timeout_minutes))
    table.add_row("Launch Mode", "[magenta]DETACHED[/magenta]" if job.detach else "[white]ATTACHED (Live Logs)[/white]")
    table.add_row("Install Requirements", "[bold green]Yes[/bold green]" if job.install_requirements else "[dim]No[/dim]")
    note = (
        "[dim]Cost estimate uses Modal public GPU pricing and the timeout limit; "
        "actual billing may include CPU, memory, storage, credits, or discounts.[/dim]"
    )
    console.print(Panel(Group(table, note), title="[bold yellow]PRE-FLIGHT CHECK[/bold yellow]", border_style="yellow"))


def nearby_directory_hint(text: str, *, limit: int = 5) -> str:
    candidate = Path(text).expanduser()
    base = candidate if candidate.is_dir() else candidate.parent
    if not str(base) or not base.exists() or not base.is_dir():
        base = Path.cwd()
    try:
        directories = sorted([p.name for p in base.iterdir() if p.is_dir()])[:limit]
    except OSError:
        return ""
    if not directories:
        return ""
    return " Available folders: " + ", ".join(directories)


def format_instruction(instruction: str | None) -> str | None:
    if not instruction:
        return None
    return f"\n  {instruction}"


def format_select_instruction(instruction: str | None) -> str | None:
    if not instruction:
        return None
    return f"\n  {instruction}"


def format_text_instruction(instruction: str | None, default: str) -> str | None:
    lines: list[str] = []
    if instruction:
        lines.append(f"  {instruction}")
    if not lines:
        return None
    return "\n" + "\n".join(lines)


def default_aware_validator(validate: Validator | None, default: str) -> Validator | None:
    if validate is None:
        return None

    def _validate(value: str) -> bool | str:
        candidate = value if value.strip() else default
        return validate(candidate)

    return _validate


def confirm_message(message: str, default: bool) -> str:
    suffix = "(Y/n)" if default else "(y/N)"
    if message.endswith(("?", ":", ".")):
        return f"{message} {suffix}"
    return f"{message} {suffix}"


def ask_text(
    message: str,
    default: str = "",
    *,
    validate: Validator | None = None,
    instruction: str | None = None,
) -> str:
    answer = questionary.text(
        message,
        default=default,
        validate=default_aware_validator(validate, default),
        instruction=format_text_instruction(instruction, default),
        style=STYLE,
    ).ask()
    if answer is None:
        raise KeyboardInterrupt
    answer = answer.strip()
    if answer:
        return answer
    return default.strip()


def ask_confirm(message: str, default: bool = False, *, instruction: str | None = None) -> bool:
    answer = questionary.confirm(
        confirm_message(message, default),
        default=default,
        instruction=format_instruction(instruction),
        style=STYLE,
    ).ask()
    if answer is None:
        raise KeyboardInterrupt
    return bool(answer)


def ask_select(
    message: str,
    choices: Sequence[str | Choice],
    default: str | Choice | None = None,
    *,
    instruction: str | None = None,
) -> Any:
    answer = questionary.select(
        message,
        choices=choices,
        default=default,
        instruction=format_select_instruction(instruction),
        show_description=True,
        style=STYLE,
    ).ask()
    if answer is None:
        raise KeyboardInterrupt
    return answer


def validate_required(label: str) -> Validator:
    def _validate(value: str) -> bool | str:
        if value.strip():
            return True
        return f"Enter a {label}."

    return _validate


def validate_positive_int(label: str, *, minimum: int = 1) -> Validator:
    def _validate(value: str) -> bool | str:
        try:
            parsed = int(value.strip())
        except ValueError:
            return f"{label} must be a whole number."
        if parsed < minimum:
            return f"{label} must be at least {minimum}."
        return True

    return _validate


def validate_positive_float(label: str) -> Validator:
    def _validate(value: str) -> bool | str:
        try:
            parsed = float(value.strip())
        except ValueError:
            return f"{label} must be a number, such as 1e-4."
        if parsed <= 0:
            return f"{label} must be greater than 0."
        return True

    return _validate


def validate_existing_dir(label: str) -> Validator:
    def _validate(value: str) -> bool | str:
        text = clean_path_input(value)
        if not text:
            return f"Enter a {label}."
        if not Path(text).expanduser().is_dir():
            return f"{label.capitalize()} not found: {text}.{nearby_directory_hint(text)}"
        return True

    return _validate


def validate_volume_name(value: str) -> bool | str:
    text = value.strip()
    if not text:
        return "Enter a Modal Volume name."
    if "/" in text or "\\" in text:
        return "Volume names cannot contain path separators."
    return True


def volume_item_is_directory(item: dict[str, Any]) -> bool:
    item_type = str(item.get("type", "")).lower()
    return item_type == "2" or item_type.endswith("directory") or item_type == "dir"


def job_directory_choices(items: Sequence[dict[str, Any]]) -> list[Choice]:
    jobs: set[str] = set()
    base = PurePosixPath("/jobs")
    candidates = [item for item in items if volume_item_is_directory(item)] or list(items)
    for item in candidates:
        raw_path = str(item.get("path", "")).strip("/")
        item_path = PurePosixPath("/" + raw_path)
        if item_path.parts[:2] != base.parts and raw_path:
            item_path = base / raw_path
        try:
            relative = item_path.relative_to(base)
        except ValueError:
            continue
        if len(relative.parts) == 1 and relative.name and safe_slug(relative.name) == relative.name:
            jobs.add(relative.name)
    return [
        Choice(job, value=job, description=f"Remove /jobs/{job}.")
        for job in sorted(jobs, key=str.casefold)
    ]


def ask_positive_int(message: str, default: str, label: str, *, minimum: int = 1) -> int:
    return int(ask_text(message, default, validate=validate_positive_int(label, minimum=minimum)))


def ask_positive_float_text(message: str, default: str, label: str) -> str:
    return ask_text(message, default, validate=validate_positive_float(label))


def clean_path_input(value: str) -> str:
    return value.strip().strip("\"'")


def ask_sanitized_name(message: str, default: str, noun: str) -> str:
    while True:
        raw_name = ask_text(
            message,
            default,
            validate=validate_required(noun),
            instruction="Allowed: letters, numbers, dots, underscores, hyphens.",
        )
        slug = safe_slug(raw_name)
        if raw_name == slug:
            return slug
        console.print(f"  [dim]Name preview:[/dim] [white]{raw_name}[/white] [dim]->[/dim] [bold cyan]{slug}[/bold cyan]")
        if ask_confirm(
            f"Use '{slug}' as the {noun} slug?",
            True,
            instruction=f"Your input will be stored as '{slug}' for Modal paths, config files, and local folders.",
        ):
            return slug


def ask_dataset_directory() -> Path:
    dataset = ask_text(
        "Local dataset directory",
        "",
        validate=validate_existing_dir("dataset directory"),
        instruction=r"Folder containing images and captions. e.g., ./data/my_images or D:\datasets\style",
    )
    return Path(clean_path_input(dataset)).expanduser()


def config_description(path: Path) -> str:
    if path.parent == JOB_CONFIG_DIR:
        return "Generated job config."
    return "Example or hand-written config."


def render_lora_config(
    job_slug: str,
    output_name: str,
    adapter: str,
    resolution: int,
    epochs: int,
    batch_size: int,
    learning_rate: str,
) -> str:
    # The generated TOML mirrors the remote /workspace layout used by Modal.
    common = f"""[Model]
base_model_path = "/workspace/Models"
trust_remote_code = true
output_dir = "/workspace/jobs/{job_slug}/output"
output_name = "{output_name}"
train_data_dir = "/workspace/jobs/{job_slug}/dataset"
resolution = {resolution}
dataloader_num_workers = 8
enable_bucket = true
use_cache = true
gemma3_prompt = "You are an assistant designed to generate high-quality anime images with the highest degree of image-text alignment based on textual prompts. <Prompt Start>"
train_batch_size = {batch_size}
num_epochs = {epochs}
save_epochs_interval = 1
learning_rate = {learning_rate}
lr_scheduler = "cosine"
lr_warmup_steps = 0
gradient_checkpointing = true
mixed_precision = "bf16"
"""

    if adapter == "LoKr":
        # LoKr keeps the upstream example's intentionally high rank/alpha defaults.
        adapter_block = """adapter_type = "lyco_lokr"
lokr_rank = 114514
lokr_alpha = 114514
lokr_train_norm = true
lokr_dropout = 0.05
lokr_rank_dropout = 0.0
lokr_module_dropout = 0.0
lokr_factor = 8
lokr_target_modules = [
    "attention.qkv",
    "attention.out",
    "feed_forward.w2",
    "time_text_embed.1",
    "clip_text_pooled_proj.1",
]
"""
    else:
        # LoRA uses smaller defaults for quicker iteration and lower memory pressure.
        adapter_block = """lora_rank = 32
lora_alpha = 32
lora_dropout = 0.05
lora_target_modules = [
    "attention.qkv",
    "attention.out",
    "feed_forward.w2",
    "time_text_embed.1",
    "clip_text_pooled_proj.1",
]
"""

    optimization = """
[Optimization]
optimizer_type = "AdamW8bit"
gradient_clip_norm = 1.0
use_flash_attention_2 = true
"""
    return common + adapter_block + optimization


def create_config_flow(*, show_steps: bool = True) -> Path:
    # Generated configs are kept separate from hand-written examples.
    JOB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if show_steps:
        print_step("Step 1: Config Identity")
    default_name = dt.datetime.now().strftime("newbie-%Y%m%d-%H%M")
    while True:
        job_slug = ask_sanitized_name("Config/job name", default_name, "config name")
        config_path = JOB_CONFIG_DIR / f"{job_slug}.toml"
        if not config_path.exists() or ask_confirm(
            f"Replace existing config '{config_path}'?",
            False,
            instruction="Choose No to enter a different config name.",
        ):
            break
    if show_steps:
        print_step("Step 2: Adapter")
    adapter = ask_select(
        "Adapter type",
        [
            Choice("LoKr", value="LoKr", checked=True, description="Recommended default for Newbie-image training quality."),
            Choice("LoRA", value="LoRA", description="Smaller adapter for quicker, lower-memory experiments."),
        ],
    )
    output_name = ask_sanitized_name("Output model name", job_slug, "output model name")
    if show_steps:
        print_step("Step 3: Training Defaults")
    resolution = int(
        ask_select(
            "Training resolution",
            [
                Choice("1024", value="1024", checked=True, description="Balanced default."),
                Choice("768", value="768", description="Lower memory and faster iterations."),
                Choice("1536", value="1536", description="Higher detail with higher GPU cost."),
            ],
        )
    )
    epochs = ask_positive_int("Epochs", "24" if adapter == "LoKr" else "50", "Epochs")
    batch_size = ask_positive_int("Batch size", "1", "Batch size")
    learning_rate = ask_positive_float_text("Learning rate", "1e-4", "Learning rate")

    config_path.write_text(
        render_lora_config(job_slug, output_name, adapter, resolution, epochs, batch_size, learning_rate),
        encoding="utf-8",
    )
    print_status(f"[bold green]Created[/bold green] {config_path}")
    return config_path


def choose_config() -> Path:
    # Offer every TOML under configs/ so examples and generated jobs share one picker.
    choices = sorted([p for p in CONFIG_DIR.rglob("*.toml") if p.is_file()])
    if not choices:
        return create_config_flow(show_steps=False)

    create_new = "__create_config__"
    labels: list[Choice] = [
        Choice(str(path), value=path, description=config_description(path))
        for path in choices
    ]
    labels.append(
        Choice(
            "Create a new config",
            value=create_new,
            description="Build a fresh LoRA/LoKr job config through guided prompts.",
        )
    )
    selected = ask_select("Training config", labels)
    if selected == create_new:
        return create_config_flow(show_steps=False)
    return selected


def run_training_flow() -> None:
    # This flow collects local choices and delegates all Modal work to the headless runner.
    print_step("Step 1: Job Identity")
    config = choose_config()
    job_name = ask_sanitized_name("Modal job name", config.stem, "job name")

    print_step("Step 2: Inputs")
    first_upload = ask_confirm(
        "Is this dataset being uploaded for this job for the first time?",
        True,
        instruction="Choose No only when both config and dataset are already in the Modal Volume for this job.",
    )
    dataset_path = ask_dataset_directory() if first_upload else None

    print_step("Step 3: Resources")
    gpu = ask_select(
        "GPU",
        [
            Choice("L40S", value="L40S", checked=True, description="Recommended default for cost and speed."),
            Choice("A100-40GB", value="A100-40GB", description="More memory for heavier runs."),
            Choice("A100-80GB", value="A100-80GB", description="Large-memory A100 option."),
            Choice("H100", value="H100", description="Fastest option with higher cost."),
            Choice("T4", value="T4", description="Lower-cost option for small tests."),
        ],
    )
    timeout = ask_positive_int("Timeout minutes", "360", "Timeout minutes")

    print_step("Step 4: Launch Options")
    install = ask_confirm(
        "Set up or update trainer dependencies before training?",
        True,
        instruction="Choose Yes for first runs or after trainer updates. Choose No only when the remote environment is already prepared.",
    )
    detach = ask_confirm(
        "Keep training running if this local terminal disconnects?",
        False,
        instruction="Detached mode submits the job and returns control locally while Modal continues running.",
    )

    job = TrainJob(
        name=job_name,
        config_path=config,
        dataset_path=dataset_path,
        gpu=gpu,
        timeout_minutes=timeout,
        install_requirements=install,
        upload=first_upload,
        detach=detach,
    )

    print_step("Step 5: Summary")
    print_training_review(job, first_upload=first_upload)
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


def delete_job_directory_flow() -> None:
    volume_name = ask_text("Volume name", DEFAULT_VOLUME, validate=validate_volume_name)
    choices = job_directory_choices(list_volume(volume_name, "/jobs"))
    if not choices:
        print_status("[dim]No job directories found in /jobs.[/dim]", style="yellow")
        return

    job = ask_select("Job directory to delete", choices)
    remote_path = f"/jobs/{job}"
    if not ask_confirm(
        f"Delete job directory '{remote_path}' from volume '{volume_name}'?",
        False,
        instruction="This removes config, dataset, logs, and outputs for this job only. It cannot be undone from the TUI.",
    ):
        return

    result = remove_job_directory(volume_name, str(job))
    print_result_panel(
        "[bold red]Job Directory Deleted[/bold red]",
        [
            ("Volume", result["volume"]),
            ("Job", result["job"]),
            ("Removed Path", result["remote_path"]),
        ],
        border_style="red",
    )


def volume_management_flow() -> None:
    while True:
        action = ask_select(
            "Volume management",
            [
                Choice("List volumes", value="list", description="Show available Modal Volumes in this account."),
                Choice("Delete a job directory", value="delete_job", description="Remove /jobs/<job> from the selected Volume."),
                Choice("Delete a volume", value="delete", description="Permanently remove a Volume and all data inside it."),
                Choice("Rename a volume", value="rename", description="Change a Volume name without downloading its contents."),
                Choice("Open dashboard", value="dashboard", description="Open the selected Volume in the Modal dashboard."),
                Choice("Back", value="back", description="Return to the main menu."),
            ],
        )
        if action == "back":
            return
        elif action == "list":
            volumes = list_all_volumes()
            if not volumes:
                print_status("[dim]No volumes found.[/dim]", style="yellow")
            else:
                table = Table(title="Modal Volumes", box=box.SIMPLE_HEAVY, padding=(0, 2))
                table.add_column("Name", style="bold white")
                table.add_column("ID", style="dim")
                for v in volumes:
                    table.add_row(str(v["name"]), str(v["id"]))
                console.print(table)
        elif action == "delete_job":
            delete_job_directory_flow()
        elif action == "delete":
            name = ask_text("Volume name to delete", DEFAULT_VOLUME, validate=validate_volume_name)
            if not ask_confirm(
                f"Delete volume '{name}' and all of its data?",
                False,
                instruction="This cannot be undone from the TUI.",
            ):
                continue
            delete_volume(name)
            print_status(f"[bold red]Deleted volume[/bold red] {name}", style="red")
        elif action == "rename":
            old_name = ask_text("Current volume name", DEFAULT_VOLUME, validate=validate_volume_name)
            new_name = ask_text("New volume name", "", validate=validate_volume_name)
            rename_volume(old_name, new_name)
            print_result_panel(
                "[bold green]Volume Renamed[/bold green]",
                [("Old Name", old_name), ("New Name", new_name)],
            )
        elif action == "dashboard":
            name = ask_text("Volume name", DEFAULT_VOLUME, validate=validate_volume_name)
            url = get_volume_dashboard_url(name)
            import webbrowser
            webbrowser.open(url)
            print_result_panel("[bold green]Dashboard Opened[/bold green]", [("URL", url)])
        console.print()


def load_model_flow() -> None:
    # Newbie training configs expect the base model to live at /workspace/Models.
    repo = ask_text(
        "Hugging Face repo or URL",
        DEFAULT_HF_REPO,
        validate=validate_required("Hugging Face repo or URL"),
        instruction="Use owner/name or a huggingface.co model URL.",
    )
    revision = ask_text("Revision", "", instruction="Leave blank to use the repository default branch.")
    timeout = ask_positive_int("Timeout minutes", "360", "Timeout minutes")
    hf_secret = ask_text(
        "Modal Secret name for HF_TOKEN",
        DEFAULT_HF_SECRET,
        validate=validate_required("Modal Secret name"),
    )
    result = download_hf_model_to_volume(repo, revision or None, DEFAULT_VOLUME, timeout, hf_secret)

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


def main() -> None:
    session_start = dt.datetime.now().astimezone()
    print_banner()
    while True:
        # Re-enter the menu after each action so operators can inspect outputs or clean up.
        action = ask_select(
            "What do you want to do?",
            [
                Choice("Run training", value="run_training", description="Upload or reuse inputs and start a Modal training job."),
                Choice("Download job output", value="download_output", description="Fetch a completed adapter folder from the Modal Volume."),
                Choice("Load model to Volume", value="load_model", description="Download the base model snapshot into /workspace/Models."),
                Choice("Create config", value="create_config", description="Generate a LoRA or LoKr TOML config for a new job."),
                Choice("Manage Volumes", value="volume_management", description="List, rename, delete, or open Modal Volumes."),
                Choice("Quit", value="quit", description="Exit without changing anything else."),
            ],
        )
        if action == "run_training":
            run_training_flow()
        elif action == "download_output":
            download_job_output_flow()
        elif action == "load_model":
            load_model_flow()
        elif action == "create_config":
            create_config_flow()
        elif action == "volume_management":
            volume_management_flow()
        else:
            print_exit_summary(session_start, dt.datetime.now().astimezone())
            return
        console.print()


if __name__ == "__main__":
    main()
