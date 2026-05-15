from __future__ import annotations

from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from scripts.training_core import DEFAULT_VOLUME, _load_modal, _load_toml_module, safe_slug


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


def remove_job_directory(volume_name: str, job_name: str) -> dict[str, str]:
    job_slug = safe_slug(job_name)
    remote_path = f"/jobs/{job_slug}"
    remove_volume_path(volume_name, remote_path, recursive=True)
    return {"volume": volume_name, "job": job_slug, "remote_path": remote_path}


def _is_volume_dir(item: Any) -> bool:
    return str(item.type) == "2" or str(item.type).lower().endswith("directory")


def download_volume_path(volume_name: str, remote_path: str, local_path: Path) -> int:
    modal = _load_modal()
    volume = modal.Volume.from_name(volume_name, create_if_missing=True)
    remote_path = "/" + remote_path.strip("/")

    def download_file(source: str, target: Path) -> int:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("wb") as file:
                return volume.read_file_into_fileobj(source, file)
        except modal.exception.NotFoundError as exc:
            raise FileNotFoundError(f"Volume path not found: {source}") from exc

    try:
        entries = volume.listdir(remote_path)
    except modal.exception.NotFoundError:
        return download_file(remote_path, local_path)

    local_path.mkdir(parents=True, exist_ok=True)
    base_path = PurePosixPath(remote_path.strip("/"))
    bytes_written = 0
    for item in entries:
        item_path = PurePosixPath(str(item.path).strip("/"))
        relative_path = Path(item_path.relative_to(base_path).as_posix())
        target_path = local_path / relative_path
        if _is_volume_dir(item):
            bytes_written += download_volume_path(volume_name, "/" + item_path.as_posix(), target_path)
        else:
            bytes_written += download_file("/" + item_path.as_posix(), target_path)
    return bytes_written


def get_config_output_name(config_path: Path) -> str:
    toml = _load_toml_module()
    config = toml.load(str(config_path.resolve()))
    output_name = str(config.get("Model", {}).get("output_name", "")).strip()
    if not output_name:
        raise ValueError(f"Config is missing Model.output_name: {config_path}")
    return output_name


def download_job_output(
    volume_name: str,
    job_name: str,
    config_path: Path,
    local_path: Path | None = None,
    output_name: str | None = None,
) -> dict[str, Any]:
    job_slug = safe_slug(job_name)
    resolved_output_name = output_name.strip() if output_name else get_config_output_name(config_path)
    if not resolved_output_name:
        raise ValueError("Output name cannot be blank.")

    remote_path = f"/jobs/{job_slug}/output/{resolved_output_name}"
    target_path = local_path or Path("outputs") / job_slug / resolved_output_name
    bytes_written = download_volume_path(volume_name, remote_path, target_path)
    return {
        "remote_path": remote_path,
        "local_path": str(target_path),
        "bytes": bytes_written,
    }


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
