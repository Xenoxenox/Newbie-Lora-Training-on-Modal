from __future__ import annotations

import datetime as dt
from pathlib import Path
import textwrap

import questionary
from questionary import Style

from modal_newbie_train import (
    DEFAULT_VOLUME,
    TrainJob,
    list_volume,
    remove_volume_path,
    run_remote_training,
    safe_slug,
)


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


def ask_text(message: str, default: str = "") -> str:
    answer = questionary.text(message, default=default, style=STYLE).ask()
    if answer is None:
        raise KeyboardInterrupt
    return answer.strip()


def ask_confirm(message: str, default: bool = False) -> bool:
    answer = questionary.confirm(message, default=default, style=STYLE).ask()
    if answer is None:
        raise KeyboardInterrupt
    return bool(answer)


def ask_select(message: str, choices: list[str], default: str | None = None) -> str:
    answer = questionary.select(message, choices=choices, default=default, style=STYLE).ask()
    if answer is None:
        raise KeyboardInterrupt
    return str(answer)


def render_lora_config(
    job_slug: str,
    output_name: str,
    adapter: str,
    resolution: int,
    epochs: int,
    batch_size: int,
    learning_rate: str,
) -> str:
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
    JOB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    default_name = dt.datetime.now().strftime("newbie-%Y%m%d-%H%M")
    job_name = ask_text("Job name", default_name)
    job_slug = safe_slug(job_name)
    adapter = ask_select("Adapter type", ["LoKr", "LoRA"], "LoKr")
    output_name = ask_text("Output model name", job_slug)
    resolution = int(ask_select("Resolution", ["1024", "768", "1536"], "1024"))
    epochs = int(ask_text("Epochs", "24" if adapter == "LoKr" else "50"))
    batch_size = int(ask_text("Batch size", "1"))
    learning_rate = ask_text("Learning rate", "1e-4")

    config_path = JOB_CONFIG_DIR / f"{job_slug}.toml"
    config_path.write_text(
        render_lora_config(job_slug, output_name, adapter, resolution, epochs, batch_size, learning_rate),
        encoding="utf-8",
    )
    print(f"\nCreated {config_path}")
    return config_path


def choose_config() -> Path:
    choices = sorted([p for p in CONFIG_DIR.rglob("*.toml") if p.is_file()])
    if not choices:
        return create_config_flow()
    labels = [str(p) for p in choices] + ["Create a new config"]
    selected = ask_select("Training config", labels)
    if selected == "Create a new config":
        return create_config_flow()
    return Path(selected)


def run_training_flow() -> None:
    config = choose_config()
    job_name = ask_text("Modal job name", config.stem)
    dataset = ask_text("Local dataset directory (blank to reuse uploaded Volume dataset)", "")
    gpu = ask_select("GPU", ["L40S", "A100-40GB", "A100-80GB", "H100", "T4"], "L40S")
    timeout = int(ask_text("Timeout minutes", "360"))
    use_xcn = ask_confirm("Use train_newbie_lora_xcn.py?", False)
    install = ask_confirm("Install trainer requirements in the remote container?", True)
    upload = ask_confirm("Upload config and dataset before running?", True)

    job = TrainJob(
        name=job_name,
        config_path=config,
        dataset_path=Path(dataset) if dataset else None,
        gpu=gpu,
        timeout_minutes=timeout,
        use_xcn_trainer=use_xcn,
        install_requirements=install,
        upload=upload,
    )
    result = run_remote_training(job)
    print("\nRemote training finished.")
    print(f"OK: {result.get('ok')}")
    print(f"Output: {result.get('output_dir')}")
    print(f"Log: {result.get('log_path')}")
    if result.get("local_zip"):
        print(f"Local zip: {result['local_zip']}")
    if not result.get("ok"):
        print("\nLog tail:\n")
        print(result.get("log_tail", ""))


def volume_browser_flow() -> None:
    path = ask_text("Volume path", "/jobs")
    items = list_volume(DEFAULT_VOLUME, path)
    if not items:
        print("\nNo entries.")
        return
    print()
    for item in items:
        size = "" if item["size"] is None else f" {item['size']} bytes"
        print(f"{item['type']:>12} {item['path']}{size}")


def cleanup_flow() -> None:
    path = ask_text("Volume path to delete", "/jobs/")
    if not path or path == "/":
        print("Refusing to delete root.")
        return
    if not ask_confirm(f"Delete {path} from Modal Volume {DEFAULT_VOLUME}?", False):
        return
    remove_volume_path(DEFAULT_VOLUME, path, recursive=True)
    print(f"Deleted {path}")


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
        action = ask_select(
            "Action",
            [
                "Run training",
                "Create config",
                "List Modal Volume",
                "Delete Volume path",
                "Quit",
            ],
        )
        if action == "Run training":
            run_training_flow()
        elif action == "Create config":
            create_config_flow()
        elif action == "List Modal Volume":
            volume_browser_flow()
        elif action == "Delete Volume path":
            cleanup_flow()
        else:
            return
        print()


if __name__ == "__main__":
    main()
