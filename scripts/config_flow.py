from __future__ import annotations

import datetime as dt
from pathlib import Path

from questionary import Choice

from modal_newbie_train import safe_slug
from scripts.preferences import load_preferences
from scripts.tui import (
    ask_confirm,
    ask_positive_float_text,
    ask_positive_int,
    ask_select,
    ask_text,
    clean_path_input,
    console,
    is_zh,
    print_status,
    print_step,
    t,
    validate_existing_dir,
    validate_required,
)


CONFIG_DIR = Path("configs")
JOB_CONFIG_DIR = CONFIG_DIR / "jobs"


def ask_sanitized_name(message: str, default: str, noun: str) -> str:
    while True:
        raw_name = ask_text(
            message,
            default,
            validate=validate_required(noun),
            instruction=t("allowed_name_chars"),
        )
        slug = safe_slug(raw_name)
        if raw_name == slug:
            return slug
        console.print(f"  [dim]{t('name_preview')}[/dim] [white]{raw_name}[/white] [dim]->[/dim] [bold cyan]{slug}[/bold cyan]")
        if ask_confirm(
            t("use_slug_confirm", slug=slug, noun=noun),
            True,
            instruction=t("use_slug_detail", slug=slug),
        ):
            return slug


def ask_dataset_directory() -> Path:
    dataset = ask_text(
        t("dataset_directory"),
        "",
        validate=validate_existing_dir("dataset directory"),
        instruction=t("dataset_folder_hint"),
    )
    return Path(clean_path_input(dataset)).expanduser()


def config_description(path: Path) -> str:
    if path.parent == JOB_CONFIG_DIR:
        return "已生成的任务配置。" if is_zh() else "Generated job config."
    return t("config_examples")


def same_config_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


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
        print_step(t("step_config_identity"))
    default_name = dt.datetime.now().strftime("newbie-%Y%m%d-%H%M")
    while True:
        job_slug = ask_sanitized_name(t("config_job_name"), default_name, "config name")
        config_path = JOB_CONFIG_DIR / f"{job_slug}.toml"
        if not config_path.exists() or ask_confirm(
            t("replace_existing_config", path=config_path),
            False,
            instruction=t("config_exists_retry"),
        ):
            break
    if show_steps:
        print_step(t("step_adapter"))
    adapter = ask_select(
        t("adapter_type"),
        [
            Choice("LoKr", value="LoKr", checked=True, description=t("adapter_lokr_desc")),
            Choice("LoRA", value="LoRA", description=t("adapter_lora_desc")),
        ],
    )
    output_name = ask_sanitized_name(t("output_model_name"), job_slug, "result folder name")
    if show_steps:
        print_step(t("step_training_defaults"))
    resolution = int(
        ask_select(
            t("training_resolution"),
            [
                Choice("1024", value="1024", checked=True, description=t("resolution_1024_desc")),
                Choice("768", value="768", description=t("resolution_768_desc")),
                Choice("1536", value="1536", description=t("resolution_1536_desc")),
            ],
        )
    )
    official_epochs = "24" if adapter == "LoKr" else "30"
    official_learning_rate = "3e-4 in the LoKr example; 1e-4 in the LoRA example"
    epochs = ask_positive_int(
        t("epochs"),
        official_epochs,
        t("epochs"),
        instruction=(
            f"Full passes over the dataset. Official example: {official_epochs} for {adapter}; "
            "more epochs raise cost and overfitting risk."
        ),
    )
    batch_size = ask_positive_int(
        t("batch_size"),
        "1",
        t("batch_size"),
        instruction=t("images_per_step"),
    )
    learning_rate = ask_positive_float_text(
        t("learning_rate"),
        "1e-4",
        t("learning_rate"),
        instruction=t("learning_rate_hint", official=official_learning_rate),
    )

    config_path.write_text(
        render_lora_config(job_slug, output_name, adapter, resolution, epochs, batch_size, learning_rate),
        encoding="utf-8",
    )
    print_status(f"[bold green]{'已创建' if is_zh() else 'Created'}[/bold green] {config_path}")
    return config_path


def choose_config(
    default_config: str | Path | None = None,
    *,
    message: str | None = None,
    instruction: str | None = None,
) -> Path:
    # Offer every TOML under configs/ so examples and generated jobs share one picker.
    choices = sorted([p for p in CONFIG_DIR.rglob("*.toml") if p.is_file()])
    if not choices:
        return create_config_flow(show_steps=False)

    default_config = default_config or load_preferences().get("last_config")
    default_path = Path(default_config) if default_config else None
    default_choice = next((path for path in choices if default_path and same_config_path(path, default_path)), None)
    create_new = "__create_config__"
    labels: list[Choice] = [
        Choice(str(path), value=path, description=config_description(path))
        for path in choices
    ]
    labels.append(
        Choice(
            t("create_new_config"),
            value=create_new,
            description=t("create_new_config_desc"),
        )
    )
    message = message or t("training_config")
    selected = ask_select(message, labels, default=default_choice, instruction=instruction)
    if selected == create_new:
        return create_config_flow(show_steps=False)
    return selected
