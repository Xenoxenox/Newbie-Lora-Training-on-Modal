from __future__ import annotations

from scripts.cli import main, parse_args
from scripts.model_ops import build_model_loader_image, download_hf_model_to_volume
from scripts.training_core import (
    APP_NAME,
    DEFAULT_HF_REPO,
    DEFAULT_HF_SECRET,
    DEFAULT_VOLUME,
    LOCAL_PYTHON_VERSION,
    REMOTE_MODELS_DIR,
    REMOTE_REPO_DIR,
    REMOTE_ROOT,
    UPSTREAM_REPO,
    VOLUME_MODELS_DIR,
    AppLogStreamer,
    TrainJob,
    _load_modal,
    _load_toml_module,
    build_image,
    build_job_config,
    dataset_upload_target,
    local_app_log_path,
    run_remote_training,
    safe_slug,
    upload_job_inputs,
)
from scripts.volume_ops import (
    _is_volume_dir,
    delete_volume,
    download_job_output,
    download_volume_path,
    get_config_output_name,
    get_volume_dashboard_url,
    list_all_volumes,
    list_volume,
    remove_job_directory,
    remove_volume_path,
    rename_volume,
)


__all__ = [
    "APP_NAME",
    "DEFAULT_HF_REPO",
    "DEFAULT_HF_SECRET",
    "DEFAULT_VOLUME",
    "LOCAL_PYTHON_VERSION",
    "REMOTE_MODELS_DIR",
    "REMOTE_REPO_DIR",
    "REMOTE_ROOT",
    "UPSTREAM_REPO",
    "VOLUME_MODELS_DIR",
    "AppLogStreamer",
    "TrainJob",
    "_is_volume_dir",
    "_load_modal",
    "_load_toml_module",
    "build_image",
    "build_job_config",
    "build_model_loader_image",
    "dataset_upload_target",
    "delete_volume",
    "download_hf_model_to_volume",
    "download_job_output",
    "download_volume_path",
    "get_config_output_name",
    "get_volume_dashboard_url",
    "list_all_volumes",
    "list_volume",
    "local_app_log_path",
    "main",
    "parse_args",
    "remove_job_directory",
    "remove_volume_path",
    "rename_volume",
    "run_remote_training",
    "safe_slug",
    "upload_job_inputs",
]


if __name__ == "__main__":
    main()
