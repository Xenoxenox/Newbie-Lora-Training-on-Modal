from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

from scripts.model_ops import download_hf_model_to_volume
from scripts.training_core import DEFAULT_HF_REPO, DEFAULT_HF_SECRET, DEFAULT_VOLUME, UPSTREAM_REPO, TrainJob, run_remote_training
from scripts.volume_ops import download_job_output, download_volume_path, list_volume, remove_volume_path


def parse_args() -> argparse.Namespace:
    # Keep CLI parsing in one place so the TUI can reuse the lower-level training functions.
    parser = argparse.ArgumentParser(
        description="Run Newbie-image LoRA training headlessly on Modal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python modal_newbie_train.py train --config configs/example_lokr.toml --dataset D:\\datasets\\my-style --job my-style
              python modal_newbie_train.py train --config configs/example_lora.toml --job resume-my-style --no-upload
              python modal_newbie_train.py train --config configs/example_lora.toml --job long-run --no-upload --detach
              python modal_newbie_train.py model-download-hf --repo NewBie-AI/NewBie-image-Exp0.1
              python modal_newbie_train.py volume-list /jobs
              python modal_newbie_train.py volume-download /jobs/<job>/output/<output-name> outputs/<job>/<output-name>
              python modal_newbie_train.py job-download --job my-style --config configs/example_lora.toml
            """
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Upload inputs and run a remote Modal training job.")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--dataset", type=Path)
    train.add_argument("--job", default="")
    train.add_argument("--gpu", default="L40S")
    train.add_argument("--timeout-minutes", type=int, default=360)
    train.add_argument("--volume", default=DEFAULT_VOLUME)
    train.add_argument("--repo-url", default=UPSTREAM_REPO)
    train.add_argument("--trainer-ref", default="main")
    train.add_argument("--no-upload", action="store_true", help="Reuse config/dataset already in the Modal Volume.")
    train.add_argument("--no-install", action="store_true", help="Skip pip install -r NewbieLoraTrainer/requirements.txt.")
    train.add_argument("--detach", action="store_true", help="Submit training and keep it running after local disconnect.")
    train.add_argument("--max-return-mb", type=int, default=128)

    volume_list = sub.add_parser("volume-list", help="List files in the Modal Volume.")
    volume_list.add_argument("path", nargs="?", default="/")
    volume_list.add_argument("--volume", default=DEFAULT_VOLUME)

    volume_rm = sub.add_parser("volume-rm", help="Remove a path from the Modal Volume.")
    volume_rm.add_argument("path")
    volume_rm.add_argument("--volume", default=DEFAULT_VOLUME)
    volume_rm.add_argument("--yes", action="store_true")

    volume_download = sub.add_parser("volume-download", help="Download a file or directory from the Modal Volume.")
    volume_download.add_argument("remote_path")
    volume_download.add_argument("local_path", type=Path)
    volume_download.add_argument("--volume", default=DEFAULT_VOLUME)

    job_download = sub.add_parser("job-download", help="Download the final adapter output for a training job.")
    job_download.add_argument("--job", required=True, help="Modal job name used for training.")
    job_download.add_argument("--config", required=True, type=Path, help="Training TOML used to get Model.output_name.")
    job_download.add_argument("--output-name", default=None, help="Override Model.output_name from the config.")
    job_download.add_argument("--local-path", type=Path, default=None, help="Local destination directory.")
    job_download.add_argument("--volume", default=DEFAULT_VOLUME)

    model_download = sub.add_parser("model-download-hf", help="Download a Hugging Face model snapshot into /workspace/Models.")
    model_download.add_argument("--repo", default=DEFAULT_HF_REPO, help="Hugging Face repo ID or https://huggingface.co/owner/name URL.")
    model_download.add_argument("--revision", default=None)
    model_download.add_argument("--volume", default=DEFAULT_VOLUME)
    model_download.add_argument("--timeout-minutes", type=int, default=360)
    model_download.add_argument("--hf-secret", default=DEFAULT_HF_SECRET, help="Modal Secret name containing an HF_TOKEN key.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        name = args.job or args.config.stem
        job = TrainJob(
            name=name,
            config_path=args.config,
            dataset_path=args.dataset,
            gpu=args.gpu,
            timeout_minutes=args.timeout_minutes,
            volume_name=args.volume,
            repo_url=args.repo_url,
            trainer_ref=args.trainer_ref,
            install_requirements=not args.no_install,
            upload=not args.no_upload,
            detach=args.detach,
            max_return_mb=args.max_return_mb,
        )
        result = run_remote_training(job)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result.get("ok") else 1)

    if args.command == "volume-list":
        for item in list_volume(args.volume, args.path):
            size = "" if item["size"] is None else f" {item['size']} bytes"
            print(f"{item['type']:>12} {item['path']}{size}")
        return

    if args.command == "volume-rm":
        if not args.yes:
            raise SystemExit("Refusing to delete without --yes.")
        remove_volume_path(args.volume, args.path)
        print(f"Removed {args.path} from {args.volume}")
        return

    if args.command == "volume-download":
        bytes_written = download_volume_path(args.volume, args.remote_path, args.local_path)
        print(f"Downloaded {args.remote_path} from {args.volume} to {args.local_path} ({bytes_written} bytes)")
        return

    if args.command == "job-download":
        result = download_job_output(args.volume, args.job, args.config, args.local_path, args.output_name)
        print(f"Downloaded {result['remote_path']} from {args.volume} to {result['local_path']} ({result['bytes']} bytes)")
        return

    if args.command == "model-download-hf":
        result = download_hf_model_to_volume(args.repo, args.revision, args.volume, args.timeout_minutes, args.hf_secret)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result.get("ok") else 1)
