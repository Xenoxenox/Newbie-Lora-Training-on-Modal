from __future__ import annotations

import argparse
import base64
import dataclasses
import datetime as dt
import json
import os
from pathlib import Path
import re
import sys
import textwrap
from typing import Any


APP_NAME = "newbie-image-lora-train"
DEFAULT_VOLUME = "newbie-image-lora"
DEFAULT_HF_SECRET = "LoRATraining"
DEFAULT_HF_REPO = "NewBie-AI/NewBie-image-Exp0.1"
REMOTE_ROOT = "/workspace"
REMOTE_MODELS_DIR = f"{REMOTE_ROOT}/Models"
VOLUME_MODELS_DIR = "/Models"
REMOTE_REPO_DIR = f"{REMOTE_ROOT}/Newbie-Lora-Trainer-Public"
UPSTREAM_REPO = "https://cnb.cool/xChenNing/Newbie-Lora-Trainer-Public.git"
LOCAL_PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"


@dataclasses.dataclass
class TrainJob:
    """Local job description that can be serialized into a Modal payload."""

    name: str
    config_path: Path
    dataset_path: Path | None
    gpu: str = "L40S"
    timeout_minutes: int = 360
    volume_name: str = DEFAULT_VOLUME
    repo_url: str = UPSTREAM_REPO
    trainer_ref: str = "main"
    use_xcn_trainer: bool = False
    install_requirements: bool = True
    upload: bool = True
    max_return_mb: int = 128

    @property
    def slug(self) -> str:
        return safe_slug(self.name)

    @property
    def volume_job_dir(self) -> str:
        return f"/jobs/{self.slug}"

    @property
    def remote_job_dir(self) -> str:
        return f"{REMOTE_ROOT}/jobs/{self.slug}"

    @property
    def volume_config(self) -> str:
        return f"{self.volume_job_dir}/config.toml"

    @property
    def remote_config(self) -> str:
        return f"{self.remote_job_dir}/config.toml"

    @property
    def volume_dataset(self) -> str:
        return f"{self.volume_job_dir}/dataset"

    @property
    def remote_dataset(self) -> str:
        return f"{self.remote_job_dir}/dataset"

    @property
    def remote_output(self) -> str:
        return f"{self.remote_job_dir}/output"

    @property
    def remote_log(self) -> str:
        return f"{self.remote_job_dir}/logs/train.log"


def safe_slug(value: str) -> str:
    # Keep job names valid for both Modal Volume paths and local output directories.
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip(".-")
    return slug or dt.datetime.now().strftime("job-%Y%m%d-%H%M%S")


def _load_modal():
    import modal  # type: ignore

    return modal


def build_image(modal: Any) -> Any:
    # Build the remote runtime image lazily so local help/list commands stay lightweight.
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
            add_python=LOCAL_PYTHON_VERSION,
        )
        .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
        .run_commands(
            "python -m pip install --upgrade pip",
            "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124",
            "pip install modal toml huggingface-hub",
        )
        .env(
            {
                "HF_HOME": f"{REMOTE_ROOT}/.cache/huggingface",
                "PIP_CACHE_DIR": f"{REMOTE_ROOT}/PipCache",
                "PYTHONUNBUFFERED": "1",
            }
        )
    )


def build_model_loader_image(modal: Any) -> Any:
    # Keep model loading separate from the heavier CUDA training image.
    return (
        modal.Image.debian_slim(python_version=LOCAL_PYTHON_VERSION)
        .pip_install("huggingface-hub>=0.23.0", "hf-transfer>=0.1.6")
        .env(
            {
                "HF_HOME": f"{REMOTE_ROOT}/.cache/huggingface",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
    )


def upload_job_inputs(job: TrainJob) -> None:
    # Upload inputs into the shared Volume before launching the remote function.
    modal = _load_modal()
    volume = modal.Volume.from_name(job.volume_name, create_if_missing=True)

    config_path = job.config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    dataset_path = job.dataset_path.resolve() if job.dataset_path else None
    if dataset_path and not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(config_path), job.volume_config)
        if dataset_path:
            batch.put_directory(str(dataset_path), job.volume_dataset)

    print(f"Uploaded config to volume:{job.volume_config} -> {job.remote_config}")
    if dataset_path:
        print(f"Uploaded dataset to volume:{job.volume_dataset} -> {job.remote_dataset}")


def download_hf_model_to_volume(
    repo_id: str,
    revision: str | None = None,
    volume_name: str = DEFAULT_VOLUME,
    timeout_minutes: int = 360,
    hf_secret_name: str = DEFAULT_HF_SECRET,
) -> dict[str, Any]:
    repo_id = repo_id.strip().removeprefix("https://huggingface.co/").strip("/")
    if "/tree/" in repo_id:
        repo_id, url_revision = repo_id.split("/tree/", 1)
        revision = revision or url_revision.strip("/") or None
    if not repo_id or "/" not in repo_id:
        raise ValueError("Hugging Face repo must be in owner/name form or a huggingface.co URL.")
    hf_secret_name = hf_secret_name.strip() or DEFAULT_HF_SECRET

    modal = _load_modal()
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    app = modal.App(f"{APP_NAME}-model-loader")
    image = build_model_loader_image(modal)

    payload = {
        "repo_id": repo_id,
        "revision": revision,
        "remote_models_dir": REMOTE_MODELS_DIR,
    }

    @app.function(
        image=image,
        timeout=timeout_minutes * 60,
        volumes={REMOTE_ROOT: volume},
        secrets=[modal.Secret.from_name(hf_secret_name)],
        serialized=True,
    )
    def modal_download_model(remote_payload: dict[str, Any]) -> dict[str, Any]:
        import os
        from pathlib import Path
        import time

        from huggingface_hub import snapshot_download

        volume.reload()

        dl_repo_id = remote_payload["repo_id"]
        dl_revision = remote_payload.get("revision") or None
        target_dir = Path(remote_payload["remote_models_dir"])
        target_dir.mkdir(parents=True, exist_ok=True)

        started = time.time()
        try:
            snapshot_download(
                repo_id=dl_repo_id,
                revision=dl_revision,
                local_dir=str(target_dir),
                token=os.environ.get("HF_TOKEN"),
            )

            files = [p for p in target_dir.rglob("*") if p.is_file()]
            total_bytes = sum(p.stat().st_size for p in files)
            result = {
                "ok": True,
                "repo_id": dl_repo_id,
                "revision": dl_revision,
                "remote_path": str(target_dir),
                "file_count": len(files),
                "bytes": total_bytes,
                "seconds": round(time.time() - started, 2),
            }
        except Exception as exc:
            result = {
                "ok": False,
                "repo_id": dl_repo_id,
                "revision": dl_revision,
                "remote_path": str(target_dir),
                "error": repr(exc),
                "seconds": round(time.time() - started, 2),
            }

        volume.commit()
        return result

    print(
        "\n正在创建 Volume 并下载模型，可能需要较长时间。\n"
        "跟踪进度：modal app list && modal app logs <app-id> -f\n"
        "或登录 https://modal.com/apps 查看。\n"
    )
    with app.run():
        return modal_download_model.remote(payload)


def run_remote_training(job: TrainJob) -> dict[str, Any]:
    # This is the public submitter used by both the CLI and the TUI.
    if job.upload:
        upload_job_inputs(job)

    modal = _load_modal()
    volume = modal.Volume.from_name(job.volume_name, create_if_missing=True)
    app = modal.App(APP_NAME)
    image = build_image(modal)

    payload = {
        "name": job.name,
        "slug": job.slug,
        "repo_url": job.repo_url,
        "trainer_ref": job.trainer_ref,
        "repo_dir": REMOTE_REPO_DIR,
        "remote_job_dir": job.remote_job_dir,
        "remote_config": job.remote_config,
        "remote_dataset": job.remote_dataset,
        "remote_output": job.remote_output,
        "remote_log": job.remote_log,
        "use_xcn_trainer": job.use_xcn_trainer,
        "install_requirements": job.install_requirements,
        "max_return_mb": job.max_return_mb,
    }

    @app.function(
        image=image,
        gpu=job.gpu,
        timeout=job.timeout_minutes * 60,
        volumes={REMOTE_ROOT: volume},
        serialized=True,
    )
    def modal_train(remote_payload: dict[str, Any]) -> dict[str, Any]:
        """Runs inside Modal; all imports are local so cloudpickle can serialize this."""

        import base64
        import os
        from pathlib import Path
        import shutil
        import subprocess
        import sys
        import time
        import zipfile

        volume.reload()

        repo_dir = Path(remote_payload["repo_dir"])
        job_dir = Path(remote_payload["remote_job_dir"])
        log_path = Path(remote_payload["remote_log"])
        output_dir = Path(remote_payload["remote_output"])
        config_file = Path(remote_payload["remote_config"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n$ {' '.join(cmd)}\n")
                log.flush()
                proc = subprocess.run(
                    cmd,
                    cwd=str(cwd) if cwd else None,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                log.write(proc.stdout)
                log.flush()
            if check and proc.returncode != 0:
                raise RuntimeError(f"Command failed with exit {proc.returncode}: {' '.join(cmd)}")
            return proc

        try:
            if repo_dir.exists() and (repo_dir / ".git").exists():
                run(["git", "fetch", "--depth", "1", "origin", remote_payload["trainer_ref"]], cwd=repo_dir, check=False)
                run(["git", "reset", "--hard", f"origin/{remote_payload['trainer_ref']}"], cwd=repo_dir, check=False)
            else:
                if repo_dir.exists():
                    shutil.rmtree(repo_dir)
                run(["git", "clone", "--depth", "1", "--branch", remote_payload["trainer_ref"], remote_payload["repo_url"], str(repo_dir)])

            model_py = repo_dir / "NewbieLoraTrainer" / "models" / "model.py"
            model_text = model_py.read_text(encoding="utf-8")
            old_import = (
                "from flash_attn import flash_attn_varlen_func\n"
                "from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa\n"
            )
            new_import = (
                "try:\n"
                "    from flash_attn import flash_attn_varlen_func\n"
                "    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa\n"
                "except Exception as exc:\n"
                "    print(f\"flash_attn unavailable, falling back to native PyTorch attention: {exc}\")\n"
                "    flash_attn_varlen_func = None\n"
                "    index_first_axis = pad_input = unpad_input = None\n"
            )
            old_condition = "if dtype in [torch.float16, torch.bfloat16] and attn_bias is None:"
            new_condition = "if flash_attn_varlen_func is not None and dtype in [torch.float16, torch.bfloat16] and attn_bias is None:"
            if old_import in model_text:
                model_text = model_text.replace(old_import, new_import)
            model_text = model_text.replace(old_condition, new_condition, 1)
            model_py.write_text(model_text, encoding="utf-8")

            requirements = repo_dir / "NewbieLoraTrainer" / "requirements.txt"
            if remote_payload["install_requirements"]:
                run([sys.executable, "-m", "pip", "install", "-U", "-r", str(requirements)])

            train_script = "train_newbie_lora_xcn.py" if remote_payload["use_xcn_trainer"] else "train_newbie_lora.py"
            command = [
                sys.executable,
                str(repo_dir / "NewbieLoraTrainer" / train_script),
                "--config_file",
                str(config_file),
            ]

            started = time.time()
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n$ {' '.join(command)}\n")
                log.flush()
                proc = subprocess.Popen(
                    command,
                    cwd=str(repo_dir),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    print(line, end="")
                    log.write(line)
                    log.flush()
                return_code = proc.wait()

            zip_path = job_dir / "output.zip"
            artifacts = []
            if output_dir.exists():
                for path in output_dir.rglob("*"):
                    if path.is_file():
                        artifacts.append(
                            {
                                "path": str(path),
                                "bytes": path.stat().st_size,
                                "relative": str(path.relative_to(output_dir)),
                            }
                        )
                if artifacts:
                    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                        for item in artifacts:
                            archive.write(item["path"], item["relative"])

            returned_zip = None
            max_bytes = int(remote_payload["max_return_mb"]) * 1024 * 1024
            if zip_path.exists() and zip_path.stat().st_size <= max_bytes:
                returned_zip = {
                    "name": zip_path.name,
                    "bytes": zip_path.stat().st_size,
                    "base64": base64.b64encode(zip_path.read_bytes()).decode("ascii"),
                }

            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            result = {
                "ok": return_code == 0,
                "return_code": return_code,
                "seconds": round(time.time() - started, 2),
                "job_dir": str(job_dir),
                "output_dir": str(output_dir),
                "log_path": str(log_path),
                "artifacts": artifacts,
                "returned_zip": returned_zip,
                "log_tail": log_text[-12000:],
            }
        except Exception as exc:
            if log_path.exists():
                log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
            else:
                log_tail = ""
            result = {
                "ok": False,
                "return_code": 999,
                "error": repr(exc),
                "job_dir": str(job_dir),
                "output_dir": str(output_dir),
                "log_path": str(log_path),
                "artifacts": [],
                "returned_zip": None,
                "log_tail": log_tail,
            }

        volume.commit()
        return result

    print(
        "\n正在启动远程训练，可能需要较长时间。\n"
        "跟踪进度：modal app list && modal app logs <app-id> -f\n"
        "或登录 https://modal.com/apps 查看。\n"
    )
    with app.run():
        result = modal_train.remote(payload)

    if result.get("returned_zip"):
        out_dir = Path("outputs") / job.slug
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_info = result["returned_zip"]
        zip_path = out_dir / zip_info["name"]
        zip_path.write_bytes(base64.b64decode(zip_info["base64"]))
        result["local_zip"] = str(zip_path)

    return result


def list_volume(volume_name: str = DEFAULT_VOLUME, path: str = "/") -> list[dict[str, Any]]:
    modal = _load_modal()
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    try:
        return [dict(path=str(item.path), type=str(item.type), size=getattr(item, "size", None)) for item in volume.listdir(path)]
    except modal.exception.NotFoundError:
        return []


def remove_volume_path(volume_name: str, path: str, recursive: bool = True) -> None:
    modal = _load_modal()
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    volume.remove_file(path, recursive=recursive)
    volume.commit()


def list_all_volumes() -> list[dict[str, str]]:
    """List all Modal volumes in the active environment."""
    modal = _load_modal()
    return [{"name": v.name, "id": v.object_id} for v in modal.Volume.objects.list()]


def delete_volume(volume_name: str) -> None:
    """Delete a named Modal volume and all of its data."""
    modal = _load_modal()
    modal.Volume.delete(volume_name)


def rename_volume(old_name: str, new_name: str) -> None:
    """Rename a Modal volume."""
    modal = _load_modal()
    modal.Volume.rename(old_name, new_name)


def get_volume_dashboard_url(volume_name: str) -> str:
    """Get the dashboard URL for a Modal volume."""
    modal = _load_modal()
    volume = modal.Volume.from_name(volume_name)
    return volume.get_dashboard_url()


def parse_args() -> argparse.Namespace:
    # Keep CLI parsing in one place so the TUI can reuse the lower-level training functions.
    parser = argparse.ArgumentParser(
        description="Run Newbie-image LoRA training headlessly on Modal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python modal_newbie_train.py train --config configs/example_lokr.toml --dataset D:\\datasets\\my-style --job my-style
              python modal_newbie_train.py train --config configs/example_lora.toml --job resume-my-style --no-upload
              python modal_newbie_train.py model-download-hf --repo NewBie-AI/NewBie-image-Exp0.1
              python modal_newbie_train.py volume-list /jobs
            """
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Upload inputs and run a remote Modal training job.")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--dataset", type=Path)
    train.add_argument("--job", default="")
    train.add_argument("--gpu", default="L40S")
    train.add_argument("--timeout-minutes", type=int, default=360)
    train.add_argument("--volume", default=DEFAULT_VOLUME)
    train.add_argument("--repo-url", default=UPSTREAM_REPO)
    train.add_argument("--trainer-ref", default="main")
    train.add_argument("--xcn", action="store_true", help="Use train_newbie_lora_xcn.py.")
    train.add_argument("--no-upload", action="store_true", help="Reuse config/dataset already in the Modal Volume.")
    train.add_argument("--no-install", action="store_true", help="Skip pip install -r NewbieLoraTrainer/requirements.txt.")
    train.add_argument("--max-return-mb", type=int, default=128)

    volume_list = sub.add_parser("volume-list", help="List files in the Modal Volume.")
    volume_list.add_argument("path", nargs="?", default="/")
    volume_list.add_argument("--volume", default=DEFAULT_VOLUME)

    volume_rm = sub.add_parser("volume-rm", help="Remove a path from the Modal Volume.")
    volume_rm.add_argument("path")
    volume_rm.add_argument("--volume", default=DEFAULT_VOLUME)
    volume_rm.add_argument("--yes", action="store_true")

    model_download = sub.add_parser("model-download-hf", help="Download a Hugging Face model snapshot into /workspace/Models.")
    model_download.add_argument("--repo", default=DEFAULT_HF_REPO, help="Hugging Face repo ID or https://huggingface.co/owner/name URL.")
    model_download.add_argument("--revision", default=None)
    model_download.add_argument("--volume", default=DEFAULT_VOLUME)
    model_download.add_argument("--timeout-minutes", type=int, default=360)
    model_download.add_argument("--hf-secret", default=DEFAULT_HF_SECRET, help="Modal Secret name containing an HF_TOKEN key.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        name = args.job or args.config.stem
        job = TrainJob(
            name=name,
            config_path=args.config,
            dataset_path=args.dataset,
            gpu=args.gpu,
            timeout_minutes=args.timeout_minutes,
            volume_name=args.volume,
            repo_url=args.repo_url,
            trainer_ref=args.trainer_ref,
            use_xcn_trainer=args.xcn,
            install_requirements=not args.no_install,
            upload=not args.no_upload,
            max_return_mb=args.max_return_mb,
        )
        result = run_remote_training(job)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result.get("ok") else 1)

    if args.command == "volume-list":
        for item in list_volume(args.volume, args.path):
            size = "" if item["size"] is None else f" {item['size']} bytes"
            print(f"{item['type']:>12} {item['path']}{size}")
        return

    if args.command == "volume-rm":
        if not args.yes:
            raise SystemExit("Refusing to delete without --yes.")
        remove_volume_path(args.volume, args.path)
        print(f"Removed {args.path} from {args.volume}")
        return

    if args.command == "model-download-hf":
        result = download_hf_model_to_volume(args.repo, args.revision, args.volume, args.timeout_minutes, args.hf_secret)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result.get("ok") else 1)

if __name__ == "__main__":
    main()
