from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import questionary
from questionary import Choice, Style
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


# Keep all TUI prompts on the same questionary theme so the menu reads as one tool.
STYLE = Style(
    [
        ("qmark", "fg:#58a6ff bold"),
        ("question", "fg:#f0f6fc bold"),
        ("answer", "fg:#7ee787 bold"),
        ("pointer", "fg:#f778ba bold"),
        ("highlighted", "fg:#f778ba bold"),
        ("selected", "fg:#7ee787"),
        ("separator", "fg:#30363d"),
        ("instruction", "fg:#8b949e italic"),
        ("text", "fg:#e6edf3"),
    ]
)

console = Console()

Validator = Callable[[str], bool | str]


def print_banner() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]NewBie-image[/bold cyan] [magenta]LoRA Training Wizard[/magenta]\n"
            "[dim]Guided Modal workflow for NewBie LoRA/LoKr jobs[/dim]",
            border_style="blue",
            padding=(1, 4),
        )
    )


def print_status(message: str, *, style: str = "green") -> None:
    console.print(Panel.fit(message, border_style=style, padding=(0, 2)))


def print_step(title: str) -> None:
    console.rule(f"[dim]{title}[/dim]", style="dim #30363d")


def print_result_panel(
    title: str,
    rows: Sequence[tuple[str, Any]],
    *,
    border_style: str = "green",
) -> None:
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim", no_wrap=True)
    table.add_column("Value", style="bold white", overflow="fold")
    for key, value in rows:
        if value is None or value == "":
            continue
        table.add_row(key, str(value))
    console.print(Panel(table, title=title, border_style=border_style, box=box.ROUNDED))


def print_log_tail(log_tail: str) -> None:
    if log_tail:
        console.print(Panel(log_tail.rstrip(), title="Log Tail", border_style="yellow"))


def clean_path_input(value: str) -> str:
    return value.strip().strip("\"'")


def nearby_directory_hint(text: str, *, limit: int = 5) -> str:
    candidate = Path(text).expanduser()
    base = candidate if candidate.is_dir() else candidate.parent
    if not str(base) or not base.exists() or not base.is_dir():
        base = Path.cwd()
    try:
        directories = sorted([p.name for p in base.iterdir() if p.is_dir()])[:limit]
    except OSError:
        return ""
    if not directories:
        return ""
    return " Available folders: " + ", ".join(directories)


def format_instruction(instruction: str | None) -> str | None:
    if not instruction:
        return None
    return f"\n  {instruction}"


def format_select_instruction(instruction: str | None) -> str | None:
    if not instruction:
        return None
    return f"\n  {instruction}"


def format_text_instruction(instruction: str | None, default: str) -> str | None:
    lines: list[str] = []
    if instruction:
        lines.append(f"  {instruction}")
    if not lines:
        return None
    return "\n" + "\n".join(lines)


def default_aware_validator(validate: Validator | None, default: str) -> Validator | None:
    if validate is None:
        return None

    def _validate(value: str) -> bool | str:
        candidate = value if value.strip() else default
        return validate(candidate)

    return _validate


def confirm_message(message: str, default: bool) -> str:
    suffix = "(Y/n)" if default else "(y/N)"
    if message.endswith(("?", ":", ".")):
        return f"{message} {suffix}"
    return f"{message} {suffix}"


def ask_text(
    message: str,
    default: str = "",
    *,
    validate: Validator | None = None,
    instruction: str | None = None,
) -> str:
    answer = questionary.text(
        message,
        default=default,
        validate=default_aware_validator(validate, default),
        instruction=format_text_instruction(instruction, default),
        style=STYLE,
    ).ask()
    if answer is None:
        raise KeyboardInterrupt
    answer = answer.strip()
    if answer:
        return answer
    return default.strip()


def ask_confirm(message: str, default: bool = False, *, instruction: str | None = None) -> bool:
    answer = questionary.confirm(
        confirm_message(message, default),
        default=default,
        instruction=format_instruction(instruction),
        style=STYLE,
    ).ask()
    if answer is None:
        raise KeyboardInterrupt
    return bool(answer)


def ask_select(
    message: str,
    choices: Sequence[str | Choice],
    default: str | Choice | None = None,
    *,
    instruction: str | None = None,
) -> Any:
    answer = questionary.select(
        message,
        choices=choices,
        default=default,
        instruction=format_select_instruction(instruction),
        show_description=True,
        style=STYLE,
    ).ask()
    if answer is None:
        raise KeyboardInterrupt
    return answer


def validate_required(label: str) -> Validator:
    def _validate(value: str) -> bool | str:
        if value.strip():
            return True
        return f"Enter a {label}."

    return _validate


def validate_positive_int(label: str, *, minimum: int = 1) -> Validator:
    def _validate(value: str) -> bool | str:
        try:
            parsed = int(value.strip())
        except ValueError:
            return f"{label} must be a whole number."
        if parsed < minimum:
            return f"{label} must be at least {minimum}."
        return True

    return _validate


def validate_positive_float(label: str) -> Validator:
    def _validate(value: str) -> bool | str:
        try:
            parsed = float(value.strip())
        except ValueError:
            return f"{label} must be a number, such as 1e-4."
        if parsed <= 0:
            return f"{label} must be greater than 0."
        return True

    return _validate


def validate_existing_dir(label: str) -> Validator:
    def _validate(value: str) -> bool | str:
        text = clean_path_input(value)
        if not text:
            return f"Enter a {label}."
        if not Path(text).expanduser().is_dir():
            return f"{label.capitalize()} not found: {text}.{nearby_directory_hint(text)}"
        return True

    return _validate


def validate_volume_name(value: str) -> bool | str:
    text = value.strip()
    if not text:
        return "Enter a Modal Volume name."
    if "/" in text or "\\" in text:
        return "Volume names cannot contain path separators."
    return True


def ask_positive_int(message: str, default: str, label: str, *, minimum: int = 1) -> int:
    return int(ask_text(message, default, validate=validate_positive_int(label, minimum=minimum)))


def ask_positive_float_text(message: str, default: str, label: str) -> str:
    return ask_text(message, default, validate=validate_positive_float(label))
