from __future__ import annotations

import datetime as dt
import subprocess
import sys

from questionary import Choice

from scripts.billing import print_exit_summary
from scripts.config_flow import create_config_flow
from scripts.training_flow import (
    configure_modal_secrets_flow,
    download_job_output_flow,
    load_model_flow,
    print_modal_secret_status,
    run_training_flow,
)
from scripts.secret_config import fresh_modal_status_snapshot, load_config, modal_auth_is_missing
from scripts.tui import ask_confirm, ask_select, console, get_language, print_banner, print_status, set_language, t
from scripts.volume_flow import volume_management_flow


def choose_language_flow() -> None:
    current = get_language()
    selected = ask_select(
        t("language_select_prompt"),
        [
            Choice("English", value="en", checked=current == "en", description="Use English for the interactive TUI."),
            Choice("中文", value="zh", checked=current == "zh", description="使用中文界面。"),
        ],
        default=current,
        instruction=t("language_select_desc"),
    )
    set_language(str(selected), persist=True)


def prompt_modal_setup_if_needed() -> None:
    config = load_config()
    snapshot = fresh_modal_status_snapshot(config)
    if not modal_auth_is_missing(snapshot):
        print_modal_secret_status(snapshot=snapshot)
        return

    print_modal_secret_status(snapshot=snapshot)
    try:
        run_setup = ask_confirm(t("modal_setup_prompt"), True)
    except KeyboardInterrupt:
        return
    if not run_setup:
        return

    print_status(f"[bold blue]{t('modal_setup_start')}[/bold blue]", style="blue")
    result = subprocess.run([sys.executable, "-m", "modal", "setup"], check=False)
    if result.returncode == 0:
        print_status(f"[bold green]{t('modal_setup_done')}[/bold green]", style="green")
    else:
        print_status(
            f"[yellow]{t('modal_setup_retry', code=result.returncode)}[/yellow]",
            style="yellow",
        )
    print_modal_secret_status(fresh=True)


def main() -> None:
    session_start = dt.datetime.now().astimezone()
    choose_language_flow()
    print_banner()
    prompt_modal_setup_if_needed()
    while True:
        # Re-enter the menu after each action so operators can inspect outputs or clean up.
        try:
            action = ask_select(
                t("main_menu"),
                [
                    Choice(t("main_run_training"), value="run_training", description=t("main_run_training_desc")),
                    Choice(t("main_create_config"), value="create_config", description=t("main_create_config_desc")),
                    Choice(t("main_load_model"), value="load_model", description=t("main_load_model_desc")),
                    Choice(t("main_download_output"), value="download_output", description=t("main_download_output_desc")),
                    Choice("Configure Modal Secrets" if get_language() == "en" else "配置 Modal Secret", value="secrets", description="Create or update Hugging Face Modal secrets." if get_language() == "en" else "创建或更新 Hugging Face Modal Secret。"),
                    Choice(t("main_manage_volumes"), value="volume_management", description=t("main_manage_volumes_desc")),
                    Choice(t("main_quit"), value="quit", description=t("main_quit_desc")),
                ],
            )
        except KeyboardInterrupt:
            break

        if action == "quit":
            break

        try:
            if action == "run_training":
                run_training_flow()
            elif action == "download_output":
                download_job_output_flow()
            elif action == "load_model":
                load_model_flow()
            elif action == "create_config":
                create_config_flow()
            elif action == "secrets":
                configure_modal_secrets_flow()
            elif action == "volume_management":
                volume_management_flow()
        except KeyboardInterrupt:
            console.print(f"[dim]{t('returned_to_menu')}[/dim]")
        console.print()
    print_exit_summary(session_start, dt.datetime.now().astimezone())


if __name__ == "__main__":
    main()
