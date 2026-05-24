from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Any

from questionary import Choice
import questionary
from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from modal_newbie_train import (
    DEFAULT_HF_REPO,
    DEFAULT_VOLUME,
    TrainJob,
    download_hf_model_to_volume,
    download_job_output,
    run_remote_training,
)
from scripts.billing import format_money
from scripts.config_flow import ask_dataset_directory, ask_sanitized_name, choose_config
from scripts.preferences import load_preferences, save_preferences
from scripts.secret_config import (
    CONFIG_PATH,
    HF_SECRET_SPEC,
    HF_TOKEN_KEY,
    MODAL_HF_SECRET_NAME_ENV,
    ModalStatusSnapshot,
    configured_hf_secret_name,
    fresh_modal_status_snapshot,
    load_config,
    modal_secret_names,
    modal_status_snapshot,
    save_config,
    set_hf_secret_config,
    upsert_modal_secret,
)
from scripts.tui import (
    STYLE,
    ask_confirm,
    ask_positive_int,
    ask_select,
    ask_text,
    console,
    is_zh,
    print_log_tail,
    print_result_panel,
    print_status,
    print_step,
    t,
    validate_required,
)


GPU_SECOND_RATES = {
    "H100": Decimal("0.001097"),
    "A100-80GB": Decimal("0.000694"),
    "A100-40GB": Decimal("0.000583"),
    "L40S": Decimal("0.000542"),
    "T4": Decimal("0.000164"),
}


def status_label(ok: Any) -> str:
    if ok is True:
        return f"[bold green]{t('success')}[/bold green]"
    if ok is False:
        return f"[bold red]{t('failed')}[/bold red]"
    return str(ok)


def dashboard_value(value: Any) -> str | None:
    if not value:
        return None
    return f"[bold cyan]{value}[/bold cyan]"


def gpu_label(gpu: str) -> str:
    if "100" in gpu:
        return f"[bold reverse red] {gpu} [/bold reverse red]"
    return f"[bold cyan]{gpu}[/bold cyan]"


def estimated_max_gpu_cost(gpu: str, timeout_minutes: int) -> str:
    rate = GPU_SECOND_RATES.get(gpu)
    if rate is None:
        return f"[dim]{t('session_billing_unavailable')}[/dim]"
    cost = rate * Decimal(timeout_minutes * 60)
    suffix = "按超时计算的 GPU 最高费用" if is_zh() else "GPU max by timeout"
    return f"{format_money(cost)} [dim]{suffix}[/dim]"


def dataset_review_value(job: TrainJob) -> str:
    if job.upload and job.dataset_path:
        return f"[bold green]{t('upload_new')}[/bold green] [dim]{job.dataset_path}[/dim]"
    detail = "复用 Modal Volume 中已有的数据集" if is_zh() else "Reusing dataset already in Modal Volume"
    return f"[dim]{t('reuse_volume')}[/dim] {detail}"


def launch_mode_label(detach: bool) -> str:
    zh = is_zh()
    if detach:
        return "[magenta]Detached（后台运行）[/magenta]" if zh else "[magenta]Detached (Background)[/magenta]"
    return "[white]Attached（实时日志）[/white]" if zh else "[white]Attached (Live Logs)[/white]"


def modal_app_cleanup_note(result: dict[str, Any]) -> str | None:
    stop_command = result.get("stop_command")
    if not stop_command:
        return None
    return f"Modal app normally closes automatically. If it lingers, run: {stop_command}"


def status_value(prefix: str, detail: str) -> str:
    return f"{prefix}: {detail}" if detail else prefix


def print_modal_status_snapshot(snapshot: ModalStatusSnapshot) -> None:
    account = snapshot.account
    if account.status == "ok":
        account_value = status_value("OK", account.detail)
    elif account.status == "missing":
        account_value = status_value("MISSING", account.detail)
    else:
        account_value = status_value("UNKNOWN", account.detail)
    account_warn = account.status != "ok"
    print_result_panel(
        "[bold yellow]Modal Account[/bold yellow]" if account_warn else "[bold green]Modal Account[/bold green]",
        [("Profile", account_value)],
        border_style="yellow" if account_warn else "green",
    )

    rows: list[tuple[str, str]] = []
    for status in snapshot.secrets:
        if status.status == "ok":
            value = f"OK: {status.detail}"
        elif status.status == "missing":
            value = f"MISSING: {status.detail}"
        elif status.status == "skipped":
            value = f"SKIPPED: {status.detail}"
        elif status.status == "disabled":
            value = status.detail
        else:
            value = f"UNKNOWN: {status.detail}"
        rows.append((status.label, value))
    warn = any(status.status in {"missing", "unknown", "skipped"} for status in snapshot.secrets)
    print_result_panel(
        "[bold yellow]Modal Secrets[/bold yellow]" if warn else "[bold green]Modal Secrets[/bold green]",
        rows,
        border_style="yellow" if warn else "green",
    )


def print_modal_secret_status(
    *,
    known_existing: set[str] | None = None,
    snapshot: ModalStatusSnapshot | None = None,
    fresh: bool = False,
) -> ModalStatusSnapshot:
    config = load_config()
    if snapshot is None:
        if fresh:
            snapshot = fresh_modal_status_snapshot(config, known_existing=known_existing)
        else:
            snapshot = modal_status_snapshot(config, known_existing=known_existing)
    print_modal_status_snapshot(snapshot)
    return snapshot


def configure_modal_secrets_flow() -> None:
    config = load_config()
    secret_names, list_error = modal_secret_names()
    current_secret = configured_hf_secret_name(config)
    if current_secret is None:
        description = t("modal_secret_disabled", env=MODAL_HF_SECRET_NAME_ENV)
    elif list_error:
        description = f"用 {HF_TOKEN_KEY} 配置 {current_secret}。" if is_zh() else f"Configure {current_secret} with key {HF_TOKEN_KEY}."
    elif current_secret in secret_names:
        description = f"用 {HF_TOKEN_KEY} 更新 {current_secret}。" if is_zh() else f"Update {current_secret} with key {HF_TOKEN_KEY}."
    else:
        description = f"用 {HF_TOKEN_KEY} 创建 {current_secret}。" if is_zh() else f"Create {current_secret} with key {HF_TOKEN_KEY}."

    action = ask_select(
        t("modal_secret_configure"),
        [
            Choice("Hugging Face", value="huggingface", description=description),
            Choice(t("back"), value="back", description=t("modal_secret_back")),
        ],
        instruction=t("modal_secret_tokens_note"),
    )
    if action == "back":
        return

    default_secret = current_secret or HF_SECRET_SPEC["default_name"]
    choices = [HF_SECRET_SPEC["default_name"], *sorted(secret_names - {HF_SECRET_SPEC["default_name"]})]
    secret_name = questionary.autocomplete(
        t("modal_secret_name"),
        choices=choices,
        default=default_secret,
        style=STYLE,
    ).ask()
    if secret_name is None:
        raise KeyboardInterrupt
    secret_name = secret_name.strip()
    if not secret_name:
        print_status(f"[yellow]{t('no_secret_name')}[/yellow]", style="yellow")
        return

    token = questionary.password(
        t("modal_secret_password", secret_name=secret_name),
        style=STYLE,
    ).ask()
    if token is None:
        raise KeyboardInterrupt
    token = token.strip()
    if not token:
        print_status(f"[yellow]{t('no_token')}[/yellow]", style="yellow")
        return

    upsert_modal_secret(secret_name, HF_TOKEN_KEY, token)
    os.environ[MODAL_HF_SECRET_NAME_ENV] = secret_name
    save_config(set_hf_secret_config(config, secret_name), CONFIG_PATH)
    print_status(f"[green]{t('modal_secret_saved', secret_name=secret_name)}[/green]", style="green")
    print_modal_secret_status(known_existing={secret_name})


def print_training_review(job: TrainJob) -> None:
    zh = is_zh()
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim", no_wrap=True)
    table.add_column("Value", style="bold white", overflow="fold")
    table.add_row("任务名称" if zh else "Job Name", f"[bold white]{job.name}[/bold white]")
    table.add_row("配置" if zh else "Config", str(job.config_path))
    table.add_row("数据集" if zh else "Dataset", dataset_review_value(job))
    table.add_row("上传" if zh else "Upload", "[bold green]Yes[/bold green]" if job.upload else "[dim]No[/dim]")
    table.add_row("GPU 类型" if zh else "GPU Type", gpu_label(job.gpu))
    table.add_row("超时" if zh else "Timeout", f"{job.timeout_minutes} mins")
    table.add_row("预计最高 GPU 成本" if zh else "Estimated Max GPU Cost", estimated_max_gpu_cost(job.gpu, job.timeout_minutes))
    table.add_row("启动模式" if zh else "Launch Mode", launch_mode_label(job.detach))
    table.add_row("依赖" if zh else "Dependencies", "[dim]Baked image[/dim]")
    note = (
        "[dim]费用估算使用 Modal 公开 GPU 价格和超时上限；实际账单可能包含 CPU、内存、存储、赠金或折扣。[/dim]"
        if zh
        else "[dim]Cost estimate uses Modal public GPU pricing and the timeout limit; "
        "actual billing may include CPU, memory, storage, credits, or discounts.[/dim]"
    )
    console.print(Panel(Group(table, note), title=f"[bold yellow]{t('preflight_check')}[/bold yellow]", border_style="yellow"))


def run_training_flow() -> None:
    # This flow collects local choices and delegates all Modal work to the headless runner.
    preferences = load_preferences()
    zh = is_zh()
    print_step(t("step_config_identity"))
    config = choose_config(preferences.get("last_config"))
    job_name = ask_sanitized_name(t("modal_job_name"), config.stem, "job name")

    print_step("步骤 2：输入" if zh else "Step 2: Inputs")
    dataset_source = ask_select(
        "数据集来源" if zh else "Dataset source",
        [
            Choice("上传本地数据集" if zh else "Upload local dataset", value="upload", description="为这个任务上传本地数据集文件夹。" if zh else "Upload a local dataset folder for this job."),
            Choice("复用 Volume 中已有数据集" if zh else "Reuse dataset already in Volume", value="reuse", description="使用 Modal Volume 中已有的 /jobs/<job>/dataset。" if zh else "Use the existing /jobs/<job>/dataset in Modal Volume."),
        ],
    )
    upload = dataset_source == "upload"
    dataset_path = ask_dataset_directory() if upload else None

    print_step("步骤 3：资源" if zh else "Step 3: Resources")
    preferred_gpu = str(preferences.get("last_gpu") or "L40S")
    gpu = ask_select(
        t("gpu"),
        [
            Choice("L40S", value="L40S", description="成本和速度的推荐默认选项。" if zh else "Recommended default for cost and speed."),
            Choice("A100-40GB", value="A100-40GB", description="适合较重任务的更多显存。" if zh else "More memory for heavier runs."),
            Choice("A100-80GB", value="A100-80GB", description="大显存 A100 选项。" if zh else "Large-memory A100 option."),
            Choice("H100", value="H100", description="最快，成本也更高。" if zh else "Fastest option with higher cost."),
            Choice("T4", value="T4", description="适合小测试的低成本选项。" if zh else "Lower-cost option for small tests."),
        ],
        default=preferred_gpu if preferred_gpu in GPU_SECOND_RATES else "L40S",
    )
    timeout_default = str(preferences.get("last_timeout") or "360")
    timeout = ask_positive_int(t("timeout_minutes"), timeout_default, t("timeout_minutes"))

    print_step(t("step_launch_options"))
    preferred_run_mode = str(preferences.get("last_run_mode") or "attached")
    run_mode = ask_select(
        "启动模式" if zh else "Launch mode",
        [
            Choice("Attached（实时日志）" if zh else "Attached (Live Logs)", value="attached", description="在当前终端持续输出 Modal 日志，直到训练结束。" if zh else "Stream Modal logs in this terminal until training finishes."),
            Choice("Detached（后台运行）" if zh else "Detached (Background)", value="detached", description="提交任务后让 Modal 在后台继续运行。" if zh else "Submit the job and let Modal continue in the background."),
        ],
        default=preferred_run_mode if preferred_run_mode in {"attached", "detached"} else "attached",
    )
    detach = run_mode == "detached"

    job = TrainJob(
        name=job_name,
        config_path=config,
        dataset_path=dataset_path,
        gpu=gpu,
        timeout_minutes=timeout,
        install_requirements=False,
        upload=upload,
        detach=detach,
    )

    print_step(t("step_summary"))
    print_training_review(job)
    if not ask_confirm(
        "这个配置正确吗？" if zh else "Is this configuration correct?",
        True,
        instruction="选择 No 会取消并返回主菜单。" if zh else "Choose No to cancel and return to the main menu.",
    ):
        print_status(
            f"[yellow]{t('training_submission_cancelled')}[/yellow]",
            style="yellow",
        )
        return

    save_preferences(
        {
            "last_config": str(config),
            "last_gpu": gpu,
            "last_timeout": timeout,
            "last_run_mode": run_mode,
        }
    )
    if detach:
        status_text = "正在向 Modal 提交训练任务..." if zh else "Submitting training job to Modal..."
        with console.status(f"[bold blue]{status_text}[/bold blue]", spinner="dots"):
            result = run_remote_training(job)
    else:
        print_status(f"[bold blue]{t('training_live_logs')}[/bold blue]", style="blue")
        result = run_remote_training(job)
    if result.get("submitted"):
        print_result_panel(
            f"[bold green]{t('training_job_submitted')}[/bold green]",
            [
                ("Status", "[bold green]SUBMITTED[/bold green]"),
                ("Detached", result.get("detached")),
                ("App ID", result.get("app_id")),
                ("Function Call ID", result.get("function_call_id")),
                ("Dashboard", dashboard_value(result.get("app_dashboard_url"))),
                ("Function Call", dashboard_value(result.get("function_call_dashboard_url"))),
                ("Logs Command", result.get("logs_command")),
                ("Stop Command", result.get("stop_command")),
                ("Output", result.get("output_dir")),
                ("TensorBoard", result.get("tensorboard_dir")),
                ("TensorBoard Volume", result.get("tensorboard_volume_path")),
                ("Log", result.get("log_path")),
                ("Note", modal_app_cleanup_note(result)),
            ],
        )
        return

    ok = result.get("ok")
    print_result_panel(
        f"[bold green]{t('remote_training_finished')}[/bold green]" if ok else f"[bold red]{t('remote_training_failed')}[/bold red]",
        [
            ("Status", status_label(ok)),
            ("Output", result.get("output_dir")),
            ("Log", result.get("log_path")),
            ("Local App Log", result.get("local_app_log_path")),
            ("App ID", result.get("app_id")),
            ("Dashboard", dashboard_value(result.get("app_dashboard_url"))),
            ("Function Call", dashboard_value(result.get("function_call_dashboard_url"))),
            ("Logs Command", result.get("logs_command")),
            ("Stop Command", result.get("stop_command")),
            ("Local Zip", result.get("local_zip")),
            ("TensorBoard", result.get("tensorboard_dir")),
            ("TensorBoard Volume", result.get("tensorboard_volume_path")),
            ("Note", modal_app_cleanup_note(result)),
        ],
        border_style="green" if ok else "red",
    )
    if not result.get("ok"):
        print_log_tail(result.get("log_tail", ""))


def load_model_flow() -> None:
    # Newbie training configs expect the base model to live at /workspace/Models.
    zh = is_zh()
    console.print(
        Panel(
            f"[dim]Model:[/dim] [bold cyan]{DEFAULT_HF_REPO}[/bold cyan]\n"
            f"[dim]{t('model_target')}:[/dim] [bold cyan]/workspace/Models[/bold cyan]",
            title=f"[bold cyan]{t('base_model')}[/bold cyan]",
            border_style="cyan",
        )
    )
    timeout = ask_positive_int(t("timeout_minutes"), "360", t("timeout_minutes"))
    default_secret = configured_hf_secret_name() or ""
    hf_secret = ask_text(
        t("hf_secret"),
        default_secret,
        validate=None if not default_secret else validate_required("Modal Secret name"),
        instruction="留空则使用当前环境变量/配置/默认值；输入 none 或 false 可禁用。" if zh else "Leave blank to use the current env/config/default; enter none or false to disable.",
    )
    result = download_hf_model_to_volume(DEFAULT_HF_REPO, None, DEFAULT_VOLUME, timeout, hf_secret)

    ok = result.get("ok")
    print_result_panel(
        f"[bold green]{t('model_load_finished')}[/bold green]" if ok else f"[bold red]{t('model_load_failed')}[/bold red]",
        [
            ("Status", status_label(ok)),
            ("Remote Path", result.get("remote_path")),
            ("Files", result.get("file_count")),
            ("Bytes", result.get("bytes")),
            ("Error", result.get("error")),
        ],
        border_style="green" if ok else "red",
    )


def download_job_output_flow() -> None:
    # Download the final adapter folder inferred from the job name and config.
    zh = is_zh()
    config = choose_config(
        message=t("config_for_output_lookup"),
        instruction=t("output_name_override_hint"),
    )
    job_name = ask_sanitized_name(t("download_job_name"), config.stem, "job name")
    output_name = ask_text(
        t("remote_output_override"),
        "",
        instruction="留空则使用所选配置中的 Model.output_name。" if zh else "Leave blank to use Model.output_name from the selected config.",
    )
    local_path = ask_text(
        t("local_destination"),
        "",
        instruction=t("local_destination_hint"),
    )

    try:
        result = download_job_output(
            DEFAULT_VOLUME,
            job_name,
            config,
            Path(local_path) if local_path else None,
            output_name or None,
        )
    except FileNotFoundError as exc:
        console.print(
            Panel(
                f"{exc}\n\n[dim]{'留空覆盖值即可使用所选配置中的 Model.output_name。' if zh else 'Leave the override blank to use Model.output_name from the selected config.'}[/dim]",
                title=f"[bold red]{t('download_failed')}[/bold red]",
                border_style="red",
            )
        )
        return

    print_result_panel(
        "[bold green]任务输出已下载[/bold green]" if zh else "[bold green]Job Output Downloaded[/bold green]",
        [
            ("Remote Path", result["remote_path"]),
            ("Local Path", result["local_path"]),
            ("Bytes", result["bytes"]),
        ],
    )
