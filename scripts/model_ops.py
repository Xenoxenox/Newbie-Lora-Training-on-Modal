from __future__ import annotations

from typing import Any

from scripts.training_core import (
    APP_NAME,
    DEFAULT_HF_SECRET,
    DEFAULT_VOLUME,
    LOCAL_PYTHON_VERSION,
    REMOTE_MODELS_DIR,
    REMOTE_ROOT,
    _load_modal,
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
        "\nCreating the Modal Volume and downloading the model. This can take a while.\n"
        "Track progress with: modal app list && modal app logs <app-id>\n"
        "You can also check https://modal.com/apps.\n"
    )
    with app.run():
        return modal_download_model.remote(payload)
