# Repository Guidelines

## Project Structure & Module Organization

This repository runs Newbie-image LoRA/LoKr training on Modal. The main headless CLI and compatibility import surface is `modal_newbie_train.py`; its implementation lives under `scripts/`: core training job types and remote launch in `scripts/training_core.py`, Hugging Face model loading in `scripts/model_ops.py`, Volume and output download operations in `scripts/volume_ops.py`, and CLI parsing/dispatch in `scripts/cli.py`. The interactive TUI entrypoint is `manage.py`. TUI implementation also lives under `scripts/`: shared prompt/render helpers in `scripts/tui.py`, config creation and selection in `scripts/config_flow.py`, training/model/output workflows in `scripts/training_flow.py`, Volume workflows in `scripts/volume_flow.py`, local non-sensitive TUI preferences in `scripts/preferences.py`, and exit billing summary in `scripts/billing.py`. Example training configs live in `configs/example_lora.toml` and `configs/example_lokr.toml`. Generated job configs are stored under `configs/jobs/` using timestamped names such as `newbie-20260507-0807.toml`. Runtime outputs, logs, virtual environments, caches, local TUI state, local `config.toml`, and zip artifacts are intentionally ignored by Git.

```text
.
|-- modal_newbie_train.py        # Headless CLI entrypoint and compatibility exports.
|-- manage.py                    # Interactive TUI entrypoint.
|-- scripts/
|   |-- cli.py                   # Headless CLI parsing and dispatch.
|   |-- training_core.py         # TrainJob, Modal training image, upload, remote launch.
|   |-- model_ops.py             # Hugging Face model download into Modal Volume.
|   |-- volume_ops.py            # Volume listing/removal/download and output download.
|   |-- tui.py                   # Shared TUI prompt, rendering, and validation helpers.
|   |-- config_flow.py           # Guided config generation and config selection.
|   |-- training_flow.py         # TUI training, model-load, and output-download flows.
|   |-- volume_flow.py           # TUI Volume management flows.
|   |-- preferences.py           # Local non-sensitive TUI preferences.
|   `-- billing.py               # Exit-time Modal billing summary.
|-- configs/
|   |-- example_lora.toml
|   |-- example_lokr.toml
|   `-- jobs/                    # Generated job configs.
|-- logs/                        # Runtime logs; ignored except .gitkeep.
|-- config.toml                  # Local non-sensitive Modal Secret names; ignored by Git.
|-- .modal-newbie/               # Local TUI preferences; ignored by Git.
`-- outputs/                     # Downloaded training artifacts; ignored by Git.
```

## Build, Test, and Development Commands

Set up a local environment with:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
modal setup
```

Run the TUI with `python manage.py`. Run headless training with:

```powershell
python modal_newbie_train.py train --config configs/example_lokr.toml --dataset D:\datasets\my-style --job my-style --gpu L40S
```

Use `python -m py_compile modal_newbie_train.py manage.py scripts/tui.py scripts/config_flow.py scripts/volume_flow.py scripts/training_flow.py scripts/billing.py scripts/training_core.py scripts/model_ops.py scripts/volume_ops.py scripts/cli.py scripts/preferences.py` as the minimum syntax check before committing Python changes that touch the TUI or Modal runner.

## Coding Style & Naming Conventions

Use idiomatic Python 3 with 4-space indentation, clear function names, and type hints where they clarify public data flow. Keep Modal-specific paths explicit and consistent with the remote `/workspace` layout. Prefer `snake_case` for functions, variables, CLI options, and TOML keys. Keep config filenames descriptive, for example `example_lokr.toml` or `my-style.toml`.

## Testing Guidelines

There is no dedicated test suite yet. For behavior changes, add focused tests if you introduce a test framework; otherwise verify with `py_compile` and a dry CLI/TUI path that avoids destructive Volume deletion. For Modal training changes, document the command used, GPU target, config file, and whether `--no-upload` or `--xcn` was used.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries, sometimes in Chinese, such as `替换README本地硬编码为仓库名` or `Initial Modal Newbie LoRA trainer`. Keep commit subjects concise and action-oriented. Pull requests should describe the user-visible change, list validation commands, call out Modal Volume or remote training impact, and include screenshots only when TUI behavior changes.

## Security & Configuration Tips

Do not commit `.env`, `config.toml`, datasets, model files, logs, outputs, local TUI state under `.modal-newbie/`, or downloaded reference repositories. Treat Modal credentials and Volume contents as private. Confirm destructive Volume operations remain guarded: TUI confirmations live in `scripts/volume_flow.py`, and CLI deletion requires `--yes` in `scripts/cli.py`. Keep TUI preferences and `config.toml` non-sensitive only; do not store Hugging Face tokens or Modal credentials there. Modal Secret names may be configured through `MODAL_HF_SECRET_NAME` or gitignored `[modal.secrets].hf_secret_name`; missing Secrets should warn and skip injection, while token values must never appear in config, docs, logs, or command panels.
