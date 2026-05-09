from __future__ import annotations

import datetime as dt
from pathlib import Path
import textwrap

import questionary
from questionary import Style

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
    # Offer every TOML under configs/ so examples and generated jobs share one picker.
    choices = sorted([p for p in CONFIG_DIR.rglob("*.toml") if p.is_file()])
    if not choices:
        return create_config_flow()
    labels = [str(p) for p in choices] + ["Create a new config"]
    selected = ask_select("Training config", labels)
    if selected == "Create a new config":
        return create_config_flow()
    return Path(selected)


def run_training_flow() -> None:
    # This flow collects local choices and delegates all Modal work to the headless runner.
    config = choose_config()
    job_name = ask_text("Modal job name", config.stem)
    dataset = ask_text("Local dataset directory (blank to reuse uploaded Volume dataset)", "")
    gpu = ask_select("GPU", ["L40S", "A100-40GB", "A100-80GB", "H100", "T4"], "L40S")
    timeout = int(ask_text("Timeout minutes", "360"))
    install = ask_confirm("Install trainer requirements in the remote container?", True)
    upload = ask_confirm("Upload config and dataset before running?", True)
    detach = ask_confirm("Detached mode (continue after local disconnect)?", False)

    job = TrainJob(
        name=job_name,
        config_path=config,
        dataset_path=Path(dataset) if dataset else None,
        gpu=gpu,
        timeout_minutes=timeout,
        install_requirements=install,
        upload=upload,
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
            ["List", "Delete", "Rename", "Dashboard", "Back"],
        )
        if action == "Back":
            return
        elif action == "List":
            volumes = list_all_volumes()
            if not volumes:
                print("\nNo volumes found.")
            else:
                print()
                for v in volumes:
                    print(f"  {v['name']:40s} {v['id']}")
        elif action == "Delete":
            name = ask_text("Volume name to delete", DEFAULT_VOLUME)
            if not ask_confirm(f"Delete volume '{name}' and ALL of its data?", False):
                continue
            delete_volume(name)
            print(f"Deleted volume '{name}'")
        elif action == "Rename":
            old_name = ask_text("Current volume name", DEFAULT_VOLUME)
            new_name = ask_text("New volume name", "")
            if not new_name:
                print("No new name provided.")
                continue
            rename_volume(old_name, new_name)
            print(f"Renamed '{old_name}' → '{new_name}'")
        elif action == "Dashboard":
            name = ask_text("Volume name", DEFAULT_VOLUME)
            url = get_volume_dashboard_url(name)
            import webbrowser
            webbrowser.open(url)
            print(f"Opened {url}")
        print()


def load_model_flow() -> None:
    # Newbie training configs expect the base model to live at /workspace/Models.
    repo = ask_text("HF repo or URL", DEFAULT_HF_REPO)
    if not repo:
        print("No repo provided.")
        return
    revision = ask_text("Revision (blank for default)", "")
    timeout = int(ask_text("Timeout minutes", "360"))
    hf_secret = ask_text("Modal Secret name for HF_TOKEN", DEFAULT_HF_SECRET)
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
    job_name = ask_text("Modal job name", config.stem)
    output_name = ask_text("Remote output folder override (blank to use config Model.output_name)", "")
    local_path = ask_text("Local destination (blank for outputs/<job>/<output>)", "")

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
            "Action",
            [
                "Run training",
                "Download job output",
                "Load model to Volume",
                "Create config",
                "Volume management",
                "Quit",
            ],
        )
        if action == "Run training":
            run_training_flow()
        elif action == "Download job output":
            download_job_output_flow()
        elif action == "Load model to Volume":
            load_model_flow()
        elif action == "Create config":
            create_config_flow()
        elif action == "Volume management":
            volume_management_flow()
        else:
            return
        print()


if __name__ == "__main__":
    main()
