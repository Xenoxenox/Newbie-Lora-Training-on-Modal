from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from scripts.training_core import DEFAULT_HF_SECRET, _load_modal, _load_toml_module


CONFIG_PATH = Path("config.toml")
MODAL_HF_SECRET_NAME_ENV = "MODAL_HF_SECRET_NAME"
HF_TOKEN_KEY = "HF_TOKEN"
DISABLED_SECRET_NAMES = {"", "none", "false"}

_secret_exists_cache: dict[str, bool] = {}
_missing_secret_warnings: set[str] = set()
TOKEN_RE = re.compile(r"\b(?:ak|as)-[A-Za-z0-9_-]+\b")


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


@dataclasses.dataclass(frozen=True)
class ModalAccountStatus:
    status: str
    detail: str
    profile: str | None = None


@dataclasses.dataclass(frozen=True)
class ModalStatusSnapshot:
    account: ModalAccountStatus
    secrets: list[ModalSecretStatus]


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
        return set(), sanitize_modal_error(str(exc))


def sanitize_modal_error(detail: str) -> str:
    return TOKEN_RE.sub("[redacted]", detail)


def modal_account_status(list_error: str | None = None) -> ModalAccountStatus:
    if list_error:
        if modal_auth_error_text(list_error):
            return ModalAccountStatus("missing", "Token missing. Run `modal setup` to sign in.")
        return ModalAccountStatus("unknown", list_error)
    profile = current_modal_profile()
    return ModalAccountStatus("ok", profile or "Authenticated", profile=profile)


def current_modal_profile() -> str | None:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "modal", "profile", "current"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    profile = result.stdout.strip()
    return profile or None


def secret_statuses_from_names(
    config: LocalConfig | None,
    secret_names: set[str],
    *,
    auth_available: bool,
    known_existing: set[str] | None = None,
) -> list[ModalSecretStatus]:
    if known_existing:
        secret_names.update(known_existing)
    secret_name = configured_hf_secret_name(config)
    if secret_name is None:
        status = "disabled"
        detail = f"disabled via {MODAL_HF_SECRET_NAME_ENV}"
    elif not auth_available:
        status = "skipped"
        detail = "Sign in to Modal before checking Hugging Face secrets."
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


def modal_secret_statuses(
    config: LocalConfig | None = None,
    *,
    known_existing: set[str] | None = None,
) -> list[ModalSecretStatus]:
    secret_names, list_error = modal_secret_names()
    return secret_statuses_from_names(
        config,
        secret_names,
        auth_available=list_error is None,
        known_existing=known_existing,
    )


def modal_auth_error_text(detail: str) -> bool:
    auth_markers = (
        "authenticate",
        "authentication",
        "credential",
        "login",
        "profile",
        "sign in",
        "signed in",
        "token",
    )
    lower_detail = detail.lower()
    return any(marker in lower_detail for marker in auth_markers)


def modal_status_snapshot(
    config: LocalConfig | None = None,
    *,
    known_existing: set[str] | None = None,
) -> ModalStatusSnapshot:
    secret_names, list_error = modal_secret_names()
    account = modal_account_status(list_error)
    secrets = secret_statuses_from_names(
        config,
        secret_names,
        auth_available=account.status == "ok",
        known_existing=known_existing,
    )
    return ModalStatusSnapshot(account=account, secrets=secrets)


def modal_auth_is_missing(snapshot_or_statuses: ModalStatusSnapshot | list[ModalSecretStatus]) -> bool:
    if isinstance(snapshot_or_statuses, ModalStatusSnapshot):
        return snapshot_or_statuses.account.status == "missing"
    return any(
        status.status in {"skipped", "unknown"} and modal_auth_error_text(status.detail)
        for status in snapshot_or_statuses
    )


def _fresh_status_probe_code(secret_name: str | None) -> str:
    auth_markers = ("authenticate", "authentication", "credential", "login", "profile", "token")
    return f"""
import json

auth_markers = {auth_markers!r}
result = {{
    "account_status": "unknown",
    "account_detail": "",
    "profile": None,
    "secret_names": [],
    "list_error": None,
}}
try:
    import modal
    try:
        from modal.config import _config_active_profile
        result["profile"] = _config_active_profile()
    except Exception:
        result["profile"] = None
    names = [secret.name for secret in modal.Secret.objects.list() if secret.name]
    result["secret_names"] = names
    result["account_status"] = "ok"
    result["account_detail"] = result["profile"] or "Authenticated"
except Exception as exc:
    detail = str(exc)
    result["list_error"] = detail
    lower_detail = detail.lower()
    result["account_status"] = "missing" if any(marker in lower_detail for marker in auth_markers) else "unknown"
    result["account_detail"] = "Token missing. Run `modal setup` to sign in." if result["account_status"] == "missing" else detail
print(json.dumps(result))
"""


def fresh_modal_status_snapshot(
    config: LocalConfig | None = None,
    *,
    known_existing: set[str] | None = None,
) -> ModalStatusSnapshot:
    config = config or load_config()
    secret_name = configured_hf_secret_name(config)
    probe = _fresh_status_probe_code(secret_name)
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        account = ModalAccountStatus("unknown", sanitize_modal_error(str(exc)))
        secrets = secret_statuses_from_names(config, set(), auth_available=False, known_existing=known_existing)
        return ModalStatusSnapshot(account=account, secrets=secrets)

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or f"status probe exited with code {result.returncode}").strip()
        account = modal_account_status(sanitize_modal_error(detail))
        secrets = secret_statuses_from_names(config, set(), auth_available=False, known_existing=known_existing)
        return ModalStatusSnapshot(account=account, secrets=secrets)

    output_lines = [line for line in result.stdout.splitlines() if line.strip()]
    output = output_lines[-1] if output_lines else ""
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        account = ModalAccountStatus("unknown", "Could not parse Modal status probe output.")
        secrets = secret_statuses_from_names(config, set(), auth_available=False, known_existing=known_existing)
        return ModalStatusSnapshot(account=account, secrets=secrets)

    account = ModalAccountStatus(
        str(parsed.get("account_status") or "unknown"),
        sanitize_modal_error(str(parsed.get("account_detail") or "")),
        profile=parsed.get("profile") if isinstance(parsed.get("profile"), str) else None,
    )
    secret_names = {name for name in parsed.get("secret_names", []) if isinstance(name, str)}
    secrets = secret_statuses_from_names(
        config,
        secret_names,
        auth_available=account.status == "ok",
        known_existing=known_existing,
    )
    return ModalStatusSnapshot(account=account, secrets=secrets)


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
