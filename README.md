# Newbie LoRA Training on Modal

Run NewBie-image LoRA/LoKr training on Modal from a local terminal.

中文版本: [README_ZH.md](README_ZH.md)

This project keeps your local machine as the controller. Base model sync, dataset upload, training, result download, and Volume maintenance run through a small TUI or a headless CLI. GPU work happens inside Modal containers; training inputs and outputs live in a Modal Volume.

## Who this is for

Use this project if you want to:

- train NewBie-image LoRA or LoKr adapters without managing a GPU server;
- keep the NewBie base model and job artifacts in a persistent Modal Volume;
- create job TOML configs from guided prompts instead of editing every field by hand;
- upload a local dataset once, then reuse the same `/jobs/<job>/dataset` path;
- run attached training with live logs or detached training that continues after local disconnect;
- download finished adapter folders back to `outputs/`.

Modal is usage-based. Check the current Modal pricing page before relying on any GPU, storage, or free-credit assumptions:

https://modal.com/pricing

## Prerequisites

- A Modal account.
- Python 3.11+ locally.
- A terminal that supports interactive prompts.
- A Hugging Face token if the base model repository requires authentication.

Install local dependencies with the one-click setup script:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

On Linux or macOS:

```bash
bash ./setup.sh
```

The script detects your operating system, installs `uv` if needed, checks network access with `ping google.com`, uses the Tsinghua PyPI mirror when that check times out, and lets `uv sync` create or update `.venv`.

If you prefer the manual path:

```powershell
uv sync
uv run modal setup
```

Use `uv run ...` for project commands so they run inside the managed environment. The setup script does not run Modal login automatically; run `uv run modal setup` when you need to authenticate Modal. The TUI also detects a missing Modal token on startup and can launch `modal setup` for you without leaving the wizard.

## Fast Path: Use The TUI

Start the guided training wizard:

```powershell
uv run python manage.py
```

On startup, choose `English` or `中文`. The choice is stored in `.modal-newbie/preferences.json` and reused on later runs. You can switch language again from the startup language prompt.

The main menu is organized around the normal training path:

- `Create Job Config` creates a LoRA/LoKr TOML under `configs/jobs/`.
- `Sync Base Model` downloads `NewBie-AI/NewBie-image-Exp0.1` into `/workspace/Models`.
- `Run Training` uploads or reuses inputs and launches a Modal training job.
- `Download Results` downloads `/jobs/<job>/output/<result-folder>`.
- `Configure Modal Secrets` creates or updates the Hugging Face Modal Secret.
- `Volume Maintenance` lists, renames, deletes, or opens Modal Volumes.

At startup, the TUI shows two status panels:

- `Modal Account`: whether the local Modal token/profile is usable.
- `Modal Secrets`: whether the configured Hugging Face Secret exists.

These are separate checks. If the Modal account token is missing, Secret checks are skipped until sign-in succeeds. When prompted, choose Yes to run `modal setup`; after the web flow finishes, the TUI refreshes the account and secret status without requiring a restart.

In submenus, `Ctrl+C` returns to the main menu. At the main menu, `Ctrl+C` exits.

## Create A Job Config

Choose `Create Job Config` in the TUI.

The generated config mirrors the Modal workspace layout:

```text
/workspace/Models
/workspace/jobs/<job>/dataset
/workspace/jobs/<job>/config.toml
/workspace/jobs/<job>/output
/workspace/jobs/<job>/output/tensorboard
```

Important prompts:

- `Config/job name`: the job slug used for Modal paths and the config filename.
- `Adapter type`: `LoKr` or `LoRA`.
- `Result folder name`: written to `Model.output_name`; `Download Results` uses it to locate `/jobs/<job>/output/<result-folder>`.
- `Training resolution`: official examples recommend `1024`; higher values cost more GPU memory.
- `Epochs`: official examples use `24` for LoKr and `30` for LoRA.
- `Batch size`: official examples use `4`; the TUI defaults to `1` for safer high-resolution Modal runs.
- `Learning rate`: official examples use `3e-4` for LoKr and `1e-4` for LoRA; upstream comments recommend testing `1e-4` or `2e-4`.

Generated configs are local files. They do not store Hugging Face tokens.

## Sync The Base Model

Choose `Sync Base Model` in the TUI before your first training run.

The TUI always uses the NewBie base model repo:

```text
NewBie-AI/NewBie-image-Exp0.1
```

It downloads the snapshot into:

```text
/workspace/Models
```

For private Hugging Face access, create a Modal Secret containing an `HF_TOKEN` key. The default Secret name is `LoRATraining`; the TUI shows it in the `Modal Secrets` panel after the `Modal Account` check succeeds. `Configure Modal Secrets` can create or update it. Enter the real token only in the password prompt or Modal CLI; do not paste token values into documentation, configs, command panels, or logs.

If the configured Secret is missing in the active Modal Environment, model sync prints a warning and runs without injecting that Secret. This keeps public model downloads from failing before the task starts. Authentication or network errors from Modal are still surfaced.

You can change the Secret name with a local environment variable:

```powershell
$env:MODAL_HF_SECRET_NAME="YourSecretName"
uv run python manage.py
```

You can also persist only the non-sensitive Secret name in gitignored `config.toml`, which is used by later TUI runs and manual CLI commands:

```toml
[modal.secrets]
hf_secret_name = "YourSecretName"
```

Local environment variables override `config.toml`, and `config.toml` overrides the default `LoRATraining` name. Set `MODAL_HF_SECRET_NAME` or `hf_secret_name` to an empty string, `none`, or `false` to disable Secret injection. Do not write token values into TOML configs, README examples, command panels, or logs.

## Prepare A Dataset

Use a local folder containing images and matching caption `.txt` files. The trainer also supports kohya-style repeated folders such as:

```text
10_character/
  image001.png
  image001.txt
```

During `Run Training`, choose one dataset source:

- `Upload local dataset`: uploads your local dataset into `/jobs/<job>/dataset`.
- `Reuse dataset already in Volume`: skips upload and expects `/jobs/<job>/dataset` to already exist.

This project intentionally keeps datasets job-scoped for now. It does not move datasets into a shared `/datasets` tree.

## Run Training

Choose `Run Training` in the TUI.

The wizard asks for:

- config;
- Modal job name;
- dataset source;
- GPU type;
- timeout minutes;
- launch mode.

Launch modes:

- `Attached (Live Logs)`: streams Modal logs locally until the run finishes.
- `Detached (Background)`: submits the job and returns control while Modal continues running.

After submission, Modal may spend time allocating GPUs and checking or building the container image before trainer logs appear. The TUI prints an explicit allocation/image status message, then prints the App ID, dashboard URL, function call, and a copyable `modal app logs ... --follow` command as soon as Modal returns them.

Before submission, the TUI shows a pre-flight summary with job name, config, dataset, upload mode, GPU, timeout, estimated max GPU cost, launch mode, and dependencies. The TUI uses the baked training image by default:

```text
Dependencies: Baked image
```

The runtime dependency install path is still available through the CLI for advanced fallback scenarios; the TUI does not prompt for it on the main path.
The baked image also installs a prebuilt FlashAttention wheel for the CUDA/PyTorch runtime, so configs with `use_flash_attention_2 = true` can use FlashAttention without a runtime compile step.

TensorBoard scalar logs are enabled by default. The runner writes event files under:

```text
/workspace/jobs/<job>/output/tensorboard
```

When a run finishes, the result panel includes the remote output path, TensorBoard path, training log path, local app log, App ID, dashboard links, log command, and a `modal app stop <app-id>` command. Modal normally closes the app automatically; use the stop command only if an app appears to linger.

## Download Results

Choose `Download Results` after a job finishes.

The config picker is used to read `Model.output_name`. Together with the Modal job name, it resolves the remote folder:

```text
/jobs/<job>/output/<output_name>
```

By default, results download to:

```text
outputs/<job>/<output_name>
```

If your remote output folder differs from `Model.output_name`, use the `Remote output folder override` prompt.

## Volume Maintenance

Choose `Volume Maintenance` in the TUI to:

- list Modal Volumes;
- delete a job directory under `/jobs/<job>`;
- delete a whole Volume;
- rename a Volume;
- open the Volume dashboard.

Destructive TUI operations require confirmation. The CLI also refuses Volume deletion unless `--yes` is passed.

The default Volume name is:

```text
newbie-image-lora
```

## Headless CLI

The TUI is recommended for normal use. The CLI is available for automation and advanced workflows.

Use `--lang zh` for Chinese CLI help and command status messages:

```powershell
uv run python modal_newbie_train.py --lang zh --help
uv run python modal_newbie_train.py --lang zh train --help
```

Sync the default base model:

```powershell
uv run python modal_newbie_train.py model-download-hf
```

Train with upload:

```powershell
uv run python modal_newbie_train.py train `
  --config configs/example_lokr.toml `
  --dataset D:\datasets\my-style `
  --job my-style `
  --gpu L40S `
  --timeout-minutes 360
```

Reuse an already uploaded job config and dataset:

```powershell
uv run python modal_newbie_train.py train `
  --config configs/example_lokr.toml `
  --job my-style `
  --no-upload
```

Submit a detached background run:

```powershell
uv run python modal_newbie_train.py train `
  --config configs/example_lora.toml `
  --job my-style `
  --no-upload `
  --detach
```

Skip runtime dependency install from the upstream trainer requirements:

```powershell
uv run python modal_newbie_train.py train `
  --config configs/example_lokr.toml `
  --dataset D:\datasets\my-style `
  --job my-style `
  --no-install
```

Download a finished job output using `Model.output_name` from the config:

```powershell
uv run python modal_newbie_train.py job-download `
  --job my-style `
  --config configs/example_lokr.toml
```

Download TensorBoard event files and view them locally:

```powershell
uv run python modal_newbie_train.py volume-download /jobs/my-style/output/tensorboard outputs/my-style/tensorboard
uv run tensorboard --logdir outputs/my-style/tensorboard
```

List and download Volume paths:

```powershell
uv run python modal_newbie_train.py volume-list /jobs
uv run python modal_newbie_train.py volume-download /jobs/my-style/output/my-style outputs/my-style/my-style
```

Remove a Volume path:

```powershell
uv run python modal_newbie_train.py volume-rm /jobs/my-style --yes
```

## Modal Workspace Layout

Modal mounts the training Volume at `/workspace`.

```text
/workspace/Newbie-Lora-Trainer-Public   # upstream trainer clone/update
/workspace/Models                       # NewBie base model snapshot
/workspace/jobs/<job>/config.toml       # uploaded job config
/workspace/jobs/<job>/dataset           # uploaded dataset
/workspace/jobs/<job>/output            # adapter outputs
/workspace/jobs/<job>/output/tensorboard # TensorBoard event files
/workspace/jobs/<job>/logs/train.log    # remote trainer log
```

The local repository stores runtime artifacts in ignored folders:

```text
logs/       # local Modal app logs
outputs/    # downloaded results
configs/jobs/ # generated job configs
.modal-newbie/preferences.json # non-sensitive TUI preferences
config.toml # non-sensitive local Modal Secret names
```

The preference file stores only non-sensitive values such as UI language, the last config, GPU, timeout, and run mode. It does not store tokens.

## Trainer Source

Remote training uses the upstream Newbie trainer:

https://cnb.cool/xChenNing/Newbie-Lora-Trainer-Public

Inside Modal, this project runs:

```bash
python /workspace/Newbie-Lora-Trainer-Public/NewbieLoraTrainer/train_newbie_lora.py --config_file /workspace/jobs/<job>/config.toml
```

The Modal training image bakes stable training dependencies. It installs the CUDA 12.4 PyTorch wheel first, then installs the matching prebuilt `flash-attn==2.7.4.post1` wheel before the remaining trainer dependencies. Runtime `install_requirements` logic remains in the headless runner as an advanced fallback, but the TUI main path uses the baked image.

The upstream trainer patch still keeps FlashAttention optional: if `flash_attn` cannot be imported, training falls back to the native PyTorch attention path instead of failing at import time.

## Security Notes

- Do not commit `.env`, datasets, model files, logs, outputs, or downloaded reference repositories.
- Treat Modal credentials and Volume contents as private.
- Keep Hugging Face tokens in Modal Secrets, not in TOML, README examples, command panels, or logs.
- Keep Modal token IDs and token secrets in Modal's local config or environment only; never paste them into issues, docs, commits, or logs.
- `config.toml` may store Modal Secret names under `[modal.secrets]`, but never token values.
- Review job names before upload: uploading a job deletes and replaces that job's old `/config.toml` and `/dataset` paths.
- Be careful with Volume names: some operations can create or target a Modal Volume by name.

## Command Reference

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
uv sync
uv run modal setup
uv run python manage.py
uv run python modal_newbie_train.py model-download-hf
uv run python modal_newbie_train.py train --config configs/example_lokr.toml --dataset D:\datasets\my-style --job my-style --gpu L40S
uv run python modal_newbie_train.py train --config configs/example_lokr.toml --job my-style --no-upload --detach
uv run python modal_newbie_train.py job-download --job my-style --config configs/example_lokr.toml
uv run python modal_newbie_train.py volume-list /jobs
```

## Contributing

Keep changes focused on the user workflow. For Python changes, run:

```powershell
uv run python -m py_compile modal_newbie_train.py manage.py scripts/tui.py scripts/config_flow.py scripts/volume_flow.py scripts/training_flow.py scripts/billing.py scripts/training_core.py scripts/model_ops.py scripts/volume_ops.py scripts/cli.py scripts/preferences.py scripts/secret_config.py
git diff --check
```

For Modal training changes, record the config, GPU, whether upload was used, and whether the run was attached or detached.
