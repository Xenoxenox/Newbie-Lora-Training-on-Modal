from __future__ import annotations

import datetime as dt

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
from scripts.tui import ask_select, console, print_banner
from scripts.volume_flow import volume_management_flow


def main() -> None:
    session_start = dt.datetime.now().astimezone()
    print_banner()
    print_modal_secret_status()
    while True:
        # Re-enter the menu after each action so operators can inspect outputs or clean up.
        try:
            action = ask_select(
                "What do you want to do?",
                [
                    Choice("Run Training", value="run_training", description="Start a NewBie LoRA/LoKr training job."),
                    Choice("Create Job Config", value="create_config", description="Generate a LoRA or LoKr TOML config for a new job."),
                    Choice("Sync Base Model", value="load_model", description="Download the NewBie base model snapshot into /workspace/Models."),
                    Choice("Download Results", value="download_output", description="Fetch a completed adapter folder from the Modal Volume."),
                    Choice("Configure Modal Secrets", value="secrets", description="Create or update Hugging Face Modal secrets."),
                    Choice("Volume Maintenance", value="volume_management", description="List, rename, delete, or open Modal Volumes."),
                    Choice("Quit", value="quit", description="Exit without changing anything else."),
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
            console.print("[dim]Returned to main menu.[/dim]")
        console.print()
    print_exit_summary(session_start, dt.datetime.now().astimezone())


if __name__ == "__main__":
    main()
