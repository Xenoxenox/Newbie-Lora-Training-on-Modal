from __future__ import annotations

from collections.abc import Callable, Sequence
import datetime as dt
from pathlib import Path
import textwrap
from typing import Any

import questionary
from questionary import Choice, Style

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
    rename_volume,
    run_remote_training,
    safe_slug,
)


# Keep all TUI prompts on the same questionary theme so the menu reads as one tool.
STYLE = Style(
    [
        ("qmark", "fg:#7ee787 bold"),
        ("question", "fg:#e6edf3 bold"),
        ("answer", "fg:#7ee787 bold"),
        ("pointer", "fg:#7ee787 bold"),
        ("highlighted", "fg:#7ee787 bold"),
        ("selected", "fg:#7ee787"),
        ("separator", "fg:#6e7681"),
        ("instruction", "fg:#8b949e"),
        ("text", "fg:#e6edf3"),
    ]
)


CONFIG_DIR = Path("configs")
JOB_CONFIG_DIR = CONFIG_DIR / "jobs"

Validator = Callable[[str], bool | str]


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
        validate=validate,
        instruction=instruction,
        style=STYLE,
    ).ask()
    if answer is None:
        raise KeyboardInterrupt
    return answer.strip()


def ask_confirm(message: str, default: bool = False, *, instruction: str | None = None) -> bool:
    answer = questionary.confirm(message, default=default, instruction=instruction, style=STYLE).ask()
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
        instruction=instruction,
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
            return f"{label.capitalize()} not found: {text}"
        return True

    return _validate


def validate_volume_name(value: str) -> bool | str:
    text = value.strip()
    if not text:
        return "Enter a Modal Volume name."
    if "/" in text or "\\" in text:
        return "Volume names cannot contain path separators."
    return True


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
            instruction="Letters, numbers, dots, underscores, and hyphens are used as-is. Other characters are converted to hyphens.",
        )
        slug = safe_slug(raw_name)
        if raw_name == slug:
            return slug
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
        instruction="Select the folder that contains the images and captions to upload for this job.",
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


def create_config_flow() -> Path:
    # Generated configs are kept separate from hand-written examples.
    JOB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
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
    adapter = ask_select(
        "Adapter type",
        [
            Choice("LoKr", value="LoKr", checked=True, description="Recommended default for Newbie-image training quality."),
            Choice("LoRA", value="LoRA", description="Smaller adapter for quicker, lower-memory experiments."),
        ],
    )
    output_name = ask_sanitized_name("Output model name", job_slug, "output model name")
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
    print(f"\nCreated {config_path}")
    return config_path


def choose_config() -> Path:
    # Offer every TOML under configs/ so examples and generated jobs share one picker.
    choices = sorted([p for p in CONFIG_DIR.rglob("*.toml") if p.is_file()])
    if not choices:
        return create_config_flow()

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
        return create_config_flow()
    return selected


def run_training_flow() -> None:
    # This flow collects local choices and delegates all Modal work to the headless runner.
    config = choose_config()
    job_name = ask_sanitized_name("Modal job name", config.stem, "job name")
    first_upload = ask_confirm(
        "Is this dataset being uploaded for this job for the first time?",
        True,
        instruction="Choose No only when both config and dataset are already in the Modal Volume for this job.",
    )
    dataset_path = ask_dataset_directory() if first_upload else None
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
    result = run_remote_training(job)
    if result.get("submitted"):
        print("\nRemote training submitted.")
        print(f"Detached: {result.get('detached')}")
        print(f"App ID: {result.get('app_id')}")
        print(f"Function call ID: {result.get('function_call_id')}")
        print(f"Dashboard: {result.get('app_dashboard_url')}")
        print(f"Function call: {result.get('function_call_dashboard_url')}")
        print(f"Output: {result.get('output_dir')}")
        print(f"Log: {result.get('log_path')}")
        return

    print("\nRemote training finished.")
    print(f"OK: {result.get('ok')}")
    print(f"Output: {result.get('output_dir')}")
    print(f"Log: {result.get('log_path')}")
    if result.get("local_zip"):
        print(f"Local zip: {result['local_zip']}")
    if not result.get("ok"):
        print("\nLog tail:\n")
        print(result.get("log_tail", ""))


def volume_management_flow() -> None:
    while True:
        action = ask_select(
            "Volume management",
            [
                Choice("List volumes", value="list", description="Show available Modal Volumes in this account."),
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
                print("\nNo volumes found.")
            else:
                print()
                for v in volumes:
                    print(f"  {v['name']:40s} {v['id']}")
        elif action == "delete":
            name = ask_text("Volume name to delete", DEFAULT_VOLUME, validate=validate_volume_name)
            if not ask_confirm(
                f"Delete volume '{name}' and all of its data?",
                False,
                instruction="This cannot be undone from the TUI.",
            ):
                continue
            delete_volume(name)
            print(f"Deleted volume '{name}'")
        elif action == "rename":
            old_name = ask_text("Current volume name", DEFAULT_VOLUME, validate=validate_volume_name)
            new_name = ask_text("New volume name", "", validate=validate_volume_name)
            rename_volume(old_name, new_name)
            print(f"Renamed '{old_name}' -> '{new_name}'")
        elif action == "dashboard":
            name = ask_text("Volume name", DEFAULT_VOLUME, validate=validate_volume_name)
            url = get_volume_dashboard_url(name)
            import webbrowser
            webbrowser.open(url)
            print(f"Opened {url}")
        print()


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

    print("\nModel load finished.")
    print(f"OK: {result.get('ok')}")
    print(f"Remote path: {result.get('remote_path')}")
    if result.get("file_count") is not None:
        print(f"Files: {result.get('file_count')}")
    if result.get("bytes") is not None:
        print(f"Bytes: {result.get('bytes')}")
    if result.get("error"):
        print(f"Error: {result['error']}")


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
        print(f"\nDownload failed: {exc}")
        print("Leave the override blank to use Model.output_name from the selected config.")
        return

    print("\nJob output downloaded.")
    print(f"Remote path: {result['remote_path']}")
    print(f"Local path: {result['local_path']}")
    print(f"Bytes: {result['bytes']}")


def main() -> None:
    print(
        textwrap.dedent(
            """
            Newbie-image LoRA Modal Manager
            -------------------------------
            """
        ).strip()
    )
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
            return
        print()


if __name__ == "__main__":
    main()
