from __future__ import annotations

import datetime as dt
from decimal import Decimal, InvalidOperation
import json
import subprocess
from typing import Any

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table

from scripts.tui import console, t


def current_modal_profile() -> str | None:
    try:
        result = subprocess.run(
            ["modal", "profile", "current"],
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


def floor_hour(value: dt.datetime) -> dt.datetime:
    return value.replace(minute=0, second=0, microsecond=0)


def ceil_hour(value: dt.datetime) -> dt.datetime:
    return floor_hour(value) + dt.timedelta(hours=1)


def modal_datetime_arg(value: dt.datetime) -> str:
    return value.isoformat(timespec="seconds")


def fetch_billing_report(start: dt.datetime, end: dt.datetime) -> list[dict[str, Any]] | None:
    try:
        result = subprocess.run(
            [
                "modal",
                "billing",
                "report",
                "--start",
                modal_datetime_arg(start),
                "--end",
                modal_datetime_arg(end),
                "--resolution",
                "h",
                "--tz",
                "local",
                "--json",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return [row for row in parsed if isinstance(row, dict)]


def parse_cost(value: Any) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def format_money(value: Decimal) -> str:
    return f"${value.quantize(Decimal('0.000001'))}"


def format_dt(value: dt.datetime) -> str:
    return value.isoformat(timespec="seconds")


def billing_summary_table(rows: list[dict[str, Any]]) -> tuple[Table, Decimal]:
    costs_by_description: dict[str, Decimal] = {}
    for row in rows:
        description = str(row.get("Description") or "Unlabeled")
        costs_by_description[description] = costs_by_description.get(description, Decimal("0")) + parse_cost(row.get("Cost"))

    total = sum(costs_by_description.values(), Decimal("0"))
    table = Table(show_header=True, box=box.SIMPLE_HEAVY, padding=(0, 2))
    table.add_column("Description", style="bold white", overflow="fold")
    table.add_column("Cost", justify="right", style="bold green", no_wrap=True)
    for description, cost in sorted(costs_by_description.items(), key=lambda item: item[1], reverse=True)[:5]:
        table.add_row(description, format_money(cost))
    return table, total


def print_exit_summary(session_start: dt.datetime, session_end: dt.datetime) -> None:
    profile = current_modal_profile()
    billing_start = floor_hour(session_start)
    billing_end = ceil_hour(session_end)
    rows = fetch_billing_report(billing_start, billing_end)

    summary = Table(show_header=False, box=None, padding=(0, 2))
    summary.add_column("Key", style="dim", no_wrap=True)
    summary.add_column("Value", style="bold white", overflow="fold")
    summary.add_row(t("modal_profile"), profile or t("session_billing_unavailable"))
    summary.add_row(t("session_start"), format_dt(session_start))
    summary.add_row(t("session_end"), format_dt(session_end))
    summary.add_row(t("billing_window"), f"{format_dt(billing_start)} -> {format_dt(billing_end)}")
    summary.add_row(t("dashboard"), "https://modal.com/apps")

    if rows is None:
        summary.add_row("Billing", f"[yellow]{t('session_billing_unavailable')}[/yellow]")
        note = f"[dim]{t('session_billing_note')}[/dim]"
        console.print(Panel(Group(summary, note), title=f"[bold blue]{t('session_closed')}[/bold blue]", border_style="blue"))
        return

    billing_table, total = billing_summary_table(rows)
    summary.add_row(t("session_total_cost"), f"[bold green]{format_money(total)}[/bold green]")
    summary.add_row(t("session_rows"), str(len(rows)))
    note = f"[dim]{t('session_billing_wait')}[/dim]"
    content = Group(
        summary,
        billing_table if rows else "[dim]No finalized billing rows for this session window yet.[/dim]",
        note,
    )
    console.print(
        Panel(
            content,
            title=f"[bold blue]{t('session_closed')}[/bold blue]",
            border_style="blue",
        )
    )
