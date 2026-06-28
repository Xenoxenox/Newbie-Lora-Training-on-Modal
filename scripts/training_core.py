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

from tui_text_zh import get_pack, normalize_lang
from scripts.preferences import load_preferences


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
REMOTE_PYTHON_VERSION = "3.11"
BAKED_TRAINER_REQUIREMENTS = (
    "modal",
    "toml>=0.10.2",
    "huggingface-hub>=0.20.0",
    "setuptools<81",
    "accelerate>=0.27.0",
    "tensorboard>=2.16.0",
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
FLASH_ATTN_RELEASE_BASE = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1"


def flash_attn_wheel_url() -> str:
    python_tag = f"cp{REMOTE_PYTHON_VERSION.replace('.', '')}"
    return (
        f"{FLASH_ATTN_RELEASE_BASE}/"
        f"flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-"
        f"{python_tag}-{python_tag}-linux_x86_64.whl"
    )


def current_text() -> Any:
    return get_pack(normalize_lang(load_preferences().get("ui_language")))


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
    def volume_tensorboard_dir(self) -> str:
        return f"{self.volume_job_dir}/output/tensorboard"

    @property
    def remote_tensorboard_dir(self) -> str:
        return f"{self.remote_output}/tensorboard"

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
            encoding="utf-8",
            errors="replace",
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


def shell_command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def modal_app_stop_command(app_id: str) -> list[str]:
    return [sys.executable, "-m", "modal", "app", "stop", app_id]


def print_modal_run_details(app_id: str, app_dashboard_url: str, function_call_id: str, function_call_dashboard_url: str) -> None:
    text = current_text()
    logs_command = shell_command_text(modal_app_logs_command(app_id, function_call_id))
    print(
        f"\n{text.app_running}\n"
        f"App ID: {app_id}\n"
        f"{text.app_dashboard}: {app_dashboard_url}\n"
        f"{text.function_call_id}: {function_call_id}\n"
        f"{text.function_call_dashboard}: {function_call_dashboard_url}\n"
        f"{text.live_logs}: {logs_command}\n",
        flush=True,
    )


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
    model_config["logging_dir"] = job.remote_tensorboard_dir
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
            add_python=REMOTE_PYTHON_VERSION,
        )
        .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
        .run_commands(
            "python -m pip install --upgrade pip",
            "pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124",
            f"pip install {shlex.quote(flash_attn_wheel_url())}",
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
        "volume_name": job.volume_name,
        "repo_url": job.repo_url,
        "trainer_ref": job.trainer_ref,
        "repo_dir": REMOTE_REPO_DIR,
        "remote_job_dir": job.remote_job_dir,
        "remote_config": job.remote_config,
        "remote_dataset": job.remote_dataset,
        "remote_output": job.remote_output,
        "volume_tensorboard_dir": job.volume_tensorboard_dir,
        "remote_tensorboard_dir": job.remote_tensorboard_dir,
        "remote_log": job.remote_log,
        "install_requirements": job.install_requirements,
        "max_return_mb": job.max_return_mb,
    }

    from scripts.remote_training import modal_train as remote_modal_train

    modal_train = app.function(
        image=image,
        gpu=job.gpu,
        timeout=job.timeout_minutes * 60,
        volumes={REMOTE_ROOT: volume},
    )(remote_modal_train)
    if job.detach:
        text = current_text()
        print(
            f"\n{text.detached_submit}\n"
            f"{text.detached_continue}\n"
            f"{text.remote_allocating}\n"
            "The app ID and log command will print after submission.\n"
            "You can also check https://modal.com/apps.\n"
        )
    else:
        text = current_text()
        print(
            f"\n{text.training_live_logs}\n"
            f"{text.synchronous_warning}\n"
            f"INFO: {text.remote_allocating}\n"
            "The app ID and log command will print when the remote function is submitted.\n"
            "You can also check https://modal.com/apps.\n"
        )

    with app.run(detach=job.detach):
        if job.detach:
            function_call = modal_train.spawn(payload)
            app_dashboard_url = app.get_dashboard_url()
            function_call_dashboard_url = function_call.get_dashboard_url()
            print_modal_run_details(app.app_id, app_dashboard_url, function_call.object_id, function_call_dashboard_url)
            result = {
                "ok": True,
                "submitted": True,
                "detached": True,
                "function_call_id": function_call.object_id,
                "function_call_dashboard_url": function_call_dashboard_url,
                "app_id": app.app_id,
                "app_dashboard_url": app_dashboard_url,
                "job_dir": f"/jobs/{job.slug}",
                "output_dir": job.remote_output,
                "tensorboard_volume_path": job.volume_tensorboard_dir,
                "tensorboard_dir": job.remote_tensorboard_dir,
                "log_path": job.remote_log,
                "logs_command": shell_command_text(modal_app_logs_command(app.app_id, function_call.object_id)),
                "stop_command": shell_command_text(modal_app_stop_command(app.app_id)),
            }
        else:
            function_call = modal_train.spawn(payload)
            app_dashboard_url = app.get_dashboard_url()
            function_call_dashboard_url = function_call.get_dashboard_url()
            print_modal_run_details(app.app_id, app_dashboard_url, function_call.object_id, function_call_dashboard_url)
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
            result["app_dashboard_url"] = app_dashboard_url
            result["function_call_id"] = function_call.object_id
            result["function_call_dashboard_url"] = function_call_dashboard_url
            result["local_app_log_path"] = str(log_streamer.local_path)
            result["logs_command"] = shell_command_text(modal_app_logs_command(app.app_id, function_call.object_id))
            result["stop_command"] = shell_command_text(modal_app_stop_command(app.app_id))
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
