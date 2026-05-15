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
    print_result_panel,
    print_status,
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
        Choice(job, value=job, description=f"Remove /jobs/{job}.")
        for job in sorted(jobs, key=str.casefold)
    ]


def delete_job_directory_flow() -> None:
    volume_name = ask_text("Volume name", DEFAULT_VOLUME, validate=validate_volume_name)
    choices = job_directory_choices(list_volume(volume_name, "/jobs"))
    if not choices:
        print_status("[dim]No job directories found in /jobs.[/dim]", style="yellow")
        return

    job = ask_select("Job directory to delete", choices)
    remote_path = f"/jobs/{job}"
    if not ask_confirm(
        f"Delete job directory '{remote_path}' from volume '{volume_name}'?",
        False,
        instruction="This removes config, dataset, logs, and outputs for this job only. It cannot be undone from the TUI.",
    ):
        return

    result = remove_job_directory(volume_name, str(job))
    print_result_panel(
        "[bold red]Job Directory Deleted[/bold red]",
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
            "Volume management",
            [
                Choice("List volumes", value="list", description="Show available Modal Volumes in this account."),
                Choice("Delete a job directory", value="delete_job", description="Remove /jobs/<job> from the selected Volume."),
                Choice("Delete a volume", value="delete", description="Permanently remove a Volume and all data inside it."),
                Choice("Rename a volume", value="rename", description="Change a Volume name without downloading its contents."),
                Choice("Open dashboard", value="dashboard", description="Open the selected Volume in the Modal dashboard."),
                Choice("Back", value="back", description="Return to the main menu."),
            ],
        )
        if action == "back":
            return
        elif action == "list":
            volumes = list_all_volumes()
            if not volumes:
                print_status("[dim]No volumes found.[/dim]", style="yellow")
            else:
                table = Table(title="Modal Volumes", box=box.SIMPLE_HEAVY, padding=(0, 2))
                table.add_column("Name", style="bold white")
                table.add_column("ID", style="dim")
                for v in volumes:
                    table.add_row(str(v["name"]), str(v["id"]))
                console.print(table)
        elif action == "delete_job":
            delete_job_directory_flow()
        elif action == "delete":
            name = ask_text("Volume name to delete", DEFAULT_VOLUME, validate=validate_volume_name)
            if not ask_confirm(
                f"Delete volume '{name}' and all of its data?",
                False,
                instruction="This cannot be undone from the TUI.",
            ):
                continue
            delete_volume(name)
            print_status(f"[bold red]Deleted volume[/bold red] {name}", style="red")
        elif action == "rename":
            old_name = ask_text("Current volume name", DEFAULT_VOLUME, validate=validate_volume_name)
            new_name = ask_text("New volume name", "", validate=validate_volume_name)
            rename_volume(old_name, new_name)
            print_result_panel(
                "[bold green]Volume Renamed[/bold green]",
                [("Old Name", old_name), ("New Name", new_name)],
            )
        elif action == "dashboard":
            name = ask_text("Volume name", DEFAULT_VOLUME, validate=validate_volume_name)
            url = get_volume_dashboard_url(name)
            webbrowser.open(url)
            print_result_panel("[bold green]Dashboard Opened[/bold green]", [("URL", url)])
        console.print()
