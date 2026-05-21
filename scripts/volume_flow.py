from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Any
import webbrowser

from questionary import Choice
from rich import box
from rich.table import Table

from modal_newbie_train import (
    DEFAULT_VOLUME,
    delete_volume,
    get_volume_dashboard_url,
    list_all_volumes,
    list_volume,
    remove_job_directory,
    rename_volume,
    safe_slug,
)
from scripts.tui import (
    ask_confirm,
    ask_select,
    ask_text,
    console,
    is_zh,
    print_result_panel,
    print_status,
    t,
    validate_volume_name,
)


def volume_item_is_directory(item: dict[str, Any]) -> bool:
    item_type = str(item.get("type", "")).lower()
    return item_type == "2" or item_type.endswith("directory") or item_type == "dir"


def job_directory_choices(items: Sequence[dict[str, Any]]) -> list[Choice]:
    jobs: set[str] = set()
    base = PurePosixPath("/jobs")
    candidates = [item for item in items if volume_item_is_directory(item)] or list(items)
    for item in candidates:
        raw_path = str(item.get("path", "")).strip("/")
        item_path = PurePosixPath("/" + raw_path)
        if item_path.parts[:2] != base.parts and raw_path:
            item_path = base / raw_path
        try:
            relative = item_path.relative_to(base)
        except ValueError:
            continue
        if len(relative.parts) == 1 and relative.name and safe_slug(relative.name) == relative.name:
            jobs.add(relative.name)
    return [
        Choice(job, value=job, description=f"删除 /jobs/{job}。" if is_zh() else f"Remove /jobs/{job}.")
        for job in sorted(jobs, key=str.casefold)
    ]


def delete_job_directory_flow() -> None:
    volume_name = ask_text(t("volume_name"), DEFAULT_VOLUME, validate=validate_volume_name)
    choices = job_directory_choices(list_volume(volume_name, "/jobs"))
    if not choices:
        print_status(f"[dim]{t('no_jobs_found')}[/dim]", style="yellow")
        return

    job = ask_select(t("job_directory_to_delete"), choices)
    remote_path = f"/jobs/{job}"
    if not ask_confirm(
        f"要从 Volume '{volume_name}' 删除任务目录 '{remote_path}' 吗？" if is_zh() else f"Delete job directory '{remote_path}' from volume '{volume_name}'?",
        False,
        instruction="这只会删除该任务的配置、数据集、日志和输出。此操作无法在 TUI 中撤销。" if is_zh() else "This removes config, dataset, logs, and outputs for this job only. It cannot be undone from the TUI.",
    ):
        return

    result = remove_job_directory(volume_name, str(job))
    print_result_panel(
        f"[bold red]{t('job_directory_deleted')}[/bold red]",
        [
            ("Volume", result["volume"]),
            ("Job", result["job"]),
            ("Removed Path", result["remote_path"]),
        ],
        border_style="red",
    )


def volume_management_flow() -> None:
    while True:
        action = ask_select(
            t("volume_management"),
            [
                Choice(t("volume_list"), value="list", description=t("volume_list_desc")),
                Choice("删除任务目录" if is_zh() else "Delete a job directory", value="delete_job", description="从所选 Volume 中删除 /jobs/<job>。" if is_zh() else "Remove /jobs/<job> from the selected Volume."),
                Choice(t("volume_delete"), value="delete", description=t("volume_delete_desc")),
                Choice(t("volume_rename"), value="rename", description=t("volume_rename_desc")),
                Choice(t("volume_dashboard"), value="dashboard", description=t("volume_dashboard_desc")),
                Choice(t("back"), value="back", description=t("modal_secret_back")),
            ],
        )
        if action == "back":
            return
        elif action == "list":
            volumes = list_all_volumes()
            if not volumes:
                print_status(f"[dim]{t('no_volumes_found')}[/dim]", style="yellow")
            else:
                table = Table(title=t("volume_list_title"), box=box.SIMPLE_HEAVY, padding=(0, 2))
                table.add_column("Name", style="bold white")
                table.add_column("ID", style="dim")
                for v in volumes:
                    table.add_row(str(v["name"]), str(v["id"]))
                console.print(table)
        elif action == "delete_job":
            delete_job_directory_flow()
        elif action == "delete":
            name = ask_text(t("volume_name_to_delete"), DEFAULT_VOLUME, validate=validate_volume_name)
            if not ask_confirm(
                f"要删除 Volume '{name}' 及其中所有数据吗？" if is_zh() else f"Delete volume '{name}' and all of its data?",
                False,
                instruction=t("volume_delete_warning"),
            ):
                continue
            delete_volume(name)
            label = "已删除 Volume" if is_zh() else "Deleted volume"
            print_status(f"[bold red]{label}[/bold red] {name}", style="red")
        elif action == "rename":
            old_name = ask_text(t("current_volume_name"), DEFAULT_VOLUME, validate=validate_volume_name)
            new_name = ask_text(t("new_volume_name"), "", validate=validate_volume_name)
            rename_volume(old_name, new_name)
            print_result_panel(
                f"[bold green]{t('volume_rename_title')}[/bold green]",
                [("Old Name", old_name), ("New Name", new_name)],
            )
        elif action == "dashboard":
            name = ask_text(t("volume_name"), DEFAULT_VOLUME, validate=validate_volume_name)
            url = get_volume_dashboard_url(name)
            webbrowser.open(url)
            print_result_panel(f"[bold green]{t('volume_dashboard_opened')}[/bold green]", [("URL", url)])
        console.print()
