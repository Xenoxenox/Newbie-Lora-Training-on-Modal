from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

from scripts.training_core import DEFAULT_HF_SECRET, _load_modal, _load_toml_module


CONFIG_PATH = Path("config.toml")
MODAL_HF_SECRET_NAME_ENV = "MODAL_HF_SECRET_NAME"
HF_TOKEN_KEY = "HF_TOKEN"
DISABLED_SECRET_NAMES = {"", "none", "false"}

_secret_exists_cache: dict[str, bool] = {}
_missing_secret_warnings: set[str] = set()


@dataclasses.dataclass(frozen=True)
class ModalSecretsConfig:
    hf_secret_name: str | None = None


@dataclasses.dataclass(frozen=True)
class LocalConfig:
    modal_secrets: ModalSecretsConfig = ModalSecretsConfig()


@dataclasses.dataclass(frozen=True)
class ModalSecretStatus:
    label: str
    name: str | None
    key: str
    status: str
    detail: str


HF_SECRET_SPEC = {
    "label": "Hugging Face",
    "env_var": MODAL_HF_SECRET_NAME_ENV,
    "default_name": DEFAULT_HF_SECRET,
    "key": HF_TOKEN_KEY,
    "config_key": "hf_secret_name",
}


def load_config(path: Path | None = None) -> LocalConfig:
    path = path or CONFIG_PATH
    if not path.exists():
        return LocalConfig()
    toml = _load_toml_module()
    raw = toml.load(str(path))
    modal = raw.get("modal", {})
    secrets = modal.get("secrets", {}) if isinstance(modal, dict) else {}
    return LocalConfig(
        modal_secrets=ModalSecretsConfig(
            hf_secret_name=secrets.get("hf_secret_name") if isinstance(secrets, dict) else None,
        )
    )


def save_config(config: LocalConfig, path: Path | None = None) -> None:
    path = path or CONFIG_PATH
    toml = _load_toml_module()
    if path.exists():
        data: dict[str, Any] = toml.load(str(path))
    else:
        data = {}
    modal_data = data.setdefault("modal", {})
    if not isinstance(modal_data, dict):
        modal_data = {}
        data["modal"] = modal_data
    secret_data = modal_data.setdefault("secrets", {})
    if not isinstance(secret_data, dict):
        secret_data = {}
        modal_data["secrets"] = secret_data
    if config.modal_secrets.hf_secret_name:
        secret_data["hf_secret_name"] = config.modal_secrets.hf_secret_name
    path.write_text(toml.dumps(data), encoding="utf-8")


def set_hf_secret_config(config: LocalConfig, secret_name: str) -> LocalConfig:
    return LocalConfig(modal_secrets=ModalSecretsConfig(hf_secret_name=secret_name))


def configured_hf_secret_name(config: LocalConfig | None = None) -> str | None:
    secret_name = os.environ.get(MODAL_HF_SECRET_NAME_ENV)
    if secret_name is None:
        if config is None:
            config = load_config()
        secret_name = config.modal_secrets.hf_secret_name
    if secret_name is None:
        secret_name = DEFAULT_HF_SECRET
    secret_name = secret_name.strip()
    if secret_name.lower() in DISABLED_SECRET_NAMES:
        return None
    return secret_name


def _modal_not_found_error(modal: Any) -> type[Exception]:
    try:
        from modal.exception import NotFoundError  # type: ignore
    except Exception:
        return getattr(modal.exception, "NotFoundError")
    return NotFoundError


def modal_secret_exists(modal: Any, secret_name: str) -> bool:
    if secret_name not in _secret_exists_cache:
        not_found_error = _modal_not_found_error(modal)
        try:
            modal.Secret.from_name(secret_name).info()
        except not_found_error:
            _secret_exists_cache[secret_name] = False
        else:
            _secret_exists_cache[secret_name] = True
    return _secret_exists_cache[secret_name]


def hf_modal_secrets(modal: Any, secret_name: str | None = None) -> list[Any]:
    secret_name = secret_name.strip() if secret_name is not None else configured_hf_secret_name()
    if secret_name is None:
        return []
    if secret_name.lower() in DISABLED_SECRET_NAMES:
        return []
    if not modal_secret_exists(modal, secret_name):
        if secret_name not in _missing_secret_warnings:
            print(
                f"Warning: Modal Secret {secret_name!r} not found; "
                "model sync will run without injecting it.",
                flush=True,
            )
            _missing_secret_warnings.add(secret_name)
        return []
    return [modal.Secret.from_name(secret_name)]


def modal_secret_names() -> tuple[set[str], str | None]:
    try:
        modal = _load_modal()
    except ImportError:
        return set(), "modal is required. Install dependencies with: uv sync"
    try:
        return {secret.name for secret in modal.Secret.objects.list() if secret.name}, None
    except Exception as exc:
        return set(), str(exc)


def modal_secret_statuses(
    config: LocalConfig | None = None,
    *,
    known_existing: set[str] | None = None,
) -> list[ModalSecretStatus]:
    secret_names, list_error = modal_secret_names()
    if known_existing:
        secret_names.update(known_existing)
    secret_name = configured_hf_secret_name(config)
    if secret_name is None:
        status = "disabled"
        detail = f"disabled via {MODAL_HF_SECRET_NAME_ENV}"
    elif list_error:
        status = "unknown"
        detail = list_error
    elif secret_name in secret_names:
        status = "ok"
        detail = f"{secret_name} ({HF_TOKEN_KEY})"
    else:
        status = "missing"
        detail = f"{secret_name} ({HF_TOKEN_KEY})"
    return [
        ModalSecretStatus(
            label=HF_SECRET_SPEC["label"],
            name=secret_name,
            key=HF_TOKEN_KEY,
            status=status,
            detail=detail,
        )
    ]


def upsert_modal_secret(secret_name: str, key: str, value: str) -> None:
    modal = _load_modal()
    not_found_error = _modal_not_found_error(modal)
    try:
        secret = modal.Secret.from_name(secret_name)
        secret.info()
    except not_found_error:
        modal.Secret.objects.create(secret_name, {key: value})
    else:
        secret.update({key: value})
    _secret_exists_cache[secret_name] = True
