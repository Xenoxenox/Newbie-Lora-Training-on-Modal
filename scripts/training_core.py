from __future__ import annotations

import base64
import dataclasses
import datetime as dt
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import threading
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
BAKED_TRAINER_REQUIREMENTS = (
    "modal",
    "toml>=0.10.2",
    "huggingface-hub>=0.20.0",
    "accelerate>=0.27.0",
    "diffusers>=0.27.0",
    "transformers>=4.38.0,<5",
    "safetensors>=0.4.0",
    "peft>=0.8.2",
    "bitsandbytes>=0.42.0",
    "lycoris-lora>=3.4.0",
    "torchdiffeq>=0.2.0",
    "timm",
    "Pillow>=10.2.0",
    "opencv-python-headless",
    "tqdm",
    "sentencepiece",
    "protobuf",
)


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
    install_requirements: bool = True
    upload: bool = True
    detach: bool = False
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


def _load_toml_module():
    import toml  # type: ignore

    return toml


def local_app_log_path(job_slug: str) -> Path:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return logs_dir / f"modal_app_{job_slug}_{timestamp}.log"


def modal_app_logs_command(app_id: str, function_call_id: str) -> list[str]:
    cmd = [sys.executable, "-m", "modal", "app", "logs"]
    try:
        result = subprocess.run(
            [*cmd, "-h"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return [*cmd, app_id]

    help_text = f"{result.stdout}\n{result.stderr}"
    if "--follow" in help_text:
        cmd.append("--follow")
    if "--function-call" in help_text:
        cmd.extend(["--function-call", function_call_id])
    cmd.append(app_id)
    return cmd


class AppLogStreamer:
    """Streams Modal App logs to stdout while teeing them into a local file."""

    def __init__(self, app_id: str, function_call_id: str, local_path: Path) -> None:
        self.app_id = app_id
        self.function_call_id = function_call_id
        self.local_path = local_path
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._file: Any = None
        self._process: subprocess.Popen[str] | None = None
        self.error: str | None = None

    def start(self) -> None:
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.local_path.open("a", encoding="utf-8", errors="replace")
        self._file.write(f"# Modal App ID: {self.app_id}\n")
        self._file.write(f"# Function Call ID: {self.function_call_id}\n\n")
        self._file.flush()
        self._thread = threading.Thread(target=self._run, name="modal-app-log-streamer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def _run(self) -> None:
        try:
            cmd = modal_app_logs_command(self.app_id, self.function_call_id)
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert self._process.stdout is not None
            for line in self._process.stdout:
                if self._stop_event.is_set():
                    break
                self._write(line)
        except Exception as exc:  # Keep training result independent from log streaming.
            self.error = repr(exc)

    def _write(self, data: str) -> None:
        sys.stdout.write(data)
        sys.stdout.flush()
        if self._file is not None:
            self._file.write(data)
            self._file.flush()


def build_job_config(job: TrainJob) -> str:
    toml = _load_toml_module()
    config_path = job.config_path.resolve()
    config = toml.load(str(config_path))
    model_config = config.setdefault("Model", {})
    model_config["train_data_dir"] = job.remote_dataset
    model_config["output_dir"] = job.remote_output
    return toml.dumps(config)


def dataset_upload_target(job: TrainJob, dataset_path: Path) -> str:
    # Preserve kohya-style repeat folders when the user selects one directly.
    if re.match(r"^\d+_.+", dataset_path.name):
        return f"{job.volume_dataset}/{dataset_path.name}"
    return job.volume_dataset


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
            "pip install " + " ".join(shlex.quote(req) for req in BAKED_TRAINER_REQUIREMENTS),
        )
        .env(
            {
                "HF_HOME": f"{REMOTE_ROOT}/.cache/huggingface",
                "PIP_CACHE_DIR": f"{REMOTE_ROOT}/PipCache",
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

    config_text = build_job_config(job)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_config = Path(temp_dir) / "config.toml"
        temp_config.write_text(config_text, encoding="utf-8")

        for path in (job.volume_config, job.volume_dataset):
            try:
                volume.remove_file(path, recursive=True)
            except Exception:
                pass

        with volume.batch_upload(force=True) as batch:
            batch.put_file(str(temp_config), job.volume_config)
            if dataset_path:
                batch.put_directory(str(dataset_path), dataset_upload_target(job, dataset_path))

    print(f"Uploaded config to volume:{job.volume_config} -> {job.remote_config}")
    if dataset_path:
        print(f"Uploaded dataset to volume:{job.volume_dataset} -> {job.remote_dataset}")


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

            trainer_py = repo_dir / "NewbieLoraTrainer" / "train_newbie_lora.py"
            trainer_text = trainer_py.read_text(encoding="utf-8")
            dtype_block = (
                "    if mixed_precision == 'bf16':\n"
                "        model_dtype = torch.bfloat16\n"
                "    elif mixed_precision == 'fp16':\n"
                "        model_dtype = torch.float16\n"
                "    else:\n"
                "        model_dtype = torch.float32\n"
            )
            patched_dtype_block = (
                "    if mixed_precision == 'bf16':\n"
                "        model_dtype = torch.bfloat16\n"
                "        clip_torch_dtype = \"bfloat16\"\n"
                "    elif mixed_precision == 'fp16':\n"
                "        model_dtype = torch.float16\n"
                "        clip_torch_dtype = \"float16\"\n"
                "    else:\n"
                "        model_dtype = torch.float32\n"
                "        clip_torch_dtype = \"float32\"\n"
            )
            if dtype_block not in trainer_text:
                raise RuntimeError(f"Unable to patch dtype block in {trainer_py}")
            trainer_text = trainer_text.replace(dtype_block, patched_dtype_block)
            clip_replacements = {
                "clip_model = AutoModel.from_pretrained(clip_model_path, torch_dtype=model_dtype, trust_remote_code=True)":
                    "clip_model = AutoModel.from_pretrained(clip_model_path, torch_dtype=clip_torch_dtype, trust_remote_code=True)",
                "clip_model = AutoModel.from_pretrained(clip_path, torch_dtype=model_dtype, trust_remote_code=True)":
                    "clip_model = AutoModel.from_pretrained(clip_path, torch_dtype=clip_torch_dtype, trust_remote_code=True)",
            }
            for old_call, new_call in clip_replacements.items():
                if old_call not in trainer_text:
                    raise RuntimeError(f"Unable to patch CLIP dtype call in {trainer_py}")
                trainer_text = trainer_text.replace(old_call, new_call)
            trainer_py.write_text(trainer_text, encoding="utf-8")

            requirements = repo_dir / "NewbieLoraTrainer" / "requirements.txt"
            requirements_text = requirements.read_text(encoding="utf-8")
            requirements_text = requirements_text.replace("transformers>=4.38.0", "transformers>=4.38.0,<5")
            requirements.write_text(requirements_text, encoding="utf-8")
            if remote_payload["install_requirements"]:
                run([sys.executable, "-m", "pip", "install", "-U", "-r", str(requirements)])

            command = [
                sys.executable,
                str(trainer_py),
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
                stream_training_output = True
                for line in proc.stdout:
                    if stream_training_output:
                        try:
                            print(line, end="")
                        except BrokenPipeError:
                            stream_training_output = False
                    log.write(line)
                    log.flush()
                return_code = proc.wait()
                if return_code == 0:
                    tail_notice = (
                        "\nTraining process exited successfully. The Modal app will close automatically.\n"
                        "If a BrokenPipeError appears after 'Training finished', it is harmless stdout shutdown noise.\n"
                    )
                    log.write(tail_notice)
                    log.flush()
                    if stream_training_output:
                        try:
                            print(tail_notice, end="")
                        except BrokenPipeError:
                            pass

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

    if job.detach:
        print(
            "\nSubmitting remote training in detached mode.\n"
            "The Modal app will keep running after the local process exits.\n"
            "Track progress with: modal app list && modal app logs <app-id>\n"
            "You can also check https://modal.com/apps.\n"
        )
    else:
        print(
            "\nStarting remote training. This can take a while.\n"
            "This is synchronous mode; disconnecting the local process may cancel this run.\n"
            "Track progress with: modal app list && modal app logs <app-id>\n"
            "You can also check https://modal.com/apps.\n"
        )

    with app.run(detach=job.detach):
        if job.detach:
            function_call = modal_train.spawn(payload)
            result = {
                "ok": True,
                "submitted": True,
                "detached": True,
                "function_call_id": function_call.object_id,
                "function_call_dashboard_url": function_call.get_dashboard_url(),
                "app_id": app.app_id,
                "app_dashboard_url": app.get_dashboard_url(),
                "job_dir": f"/jobs/{job.slug}",
                "output_dir": job.remote_output,
                "log_path": job.remote_log,
            }
        else:
            function_call = modal_train.spawn(payload)
            log_streamer = AppLogStreamer(app.app_id, function_call.object_id, local_app_log_path(job.slug))
            log_streamer.start()
            try:
                result = function_call.get()
            except KeyboardInterrupt:
                try:
                    function_call.cancel(terminate_containers=True)
                finally:
                    raise
            finally:
                log_streamer.stop()
            result["app_id"] = app.app_id
            result["app_dashboard_url"] = app.get_dashboard_url()
            result["function_call_id"] = function_call.object_id
            result["function_call_dashboard_url"] = function_call.get_dashboard_url()
            result["local_app_log_path"] = str(log_streamer.local_path)
            if log_streamer.error:
                result["local_app_log_error"] = log_streamer.error

    if result.get("returned_zip"):
        out_dir = Path("outputs") / job.slug
        out_dir.mkdir(parents=True, exist_ok=True)
        zip_info = result["returned_zip"]
        zip_path = out_dir / zip_info["name"]
        zip_path.write_bytes(base64.b64decode(zip_info["base64"]))
        result["local_zip"] = str(zip_path)

    return result
