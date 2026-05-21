from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

from tui_text_zh import normalize_lang
from scripts.model_ops import download_hf_model_to_volume
from scripts.preferences import save_preferences
from scripts.secret_config import configured_hf_secret_name
from scripts.training_core import DEFAULT_HF_REPO, DEFAULT_VOLUME, UPSTREAM_REPO, TrainJob, run_remote_training
from scripts.volume_ops import download_job_output, download_volume_path, list_volume, remove_volume_path


def _language_from_argv() -> str:
    import sys

    for index, value in enumerate(sys.argv):
        if value == "--lang" and index + 1 < len(sys.argv):
            return normalize_lang(sys.argv[index + 1])
        if value.startswith("--lang="):
            return normalize_lang(value.split("=", 1)[1])
    return "en"


def parse_args() -> argparse.Namespace:
    # Keep CLI parsing in one place so the TUI can reuse the lower-level training functions.
    lang = _language_from_argv()
    zh = lang == "zh"
    parser = argparse.ArgumentParser(
        description="在 Modal 上以 headless 模式运行 Newbie-image LoRA 训练。" if zh else "Run Newbie-image LoRA training headlessly on Modal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            示例:
              uv run python modal_newbie_train.py --lang zh train --config configs/example_lokr.toml --dataset D:\\datasets\\my-style --job my-style
              uv run python modal_newbie_train.py --lang zh train --config configs/example_lora.toml --job resume-my-style --no-upload
              uv run python modal_newbie_train.py --lang zh train --config configs/example_lora.toml --job long-run --no-upload --detach
              uv run python modal_newbie_train.py --lang zh model-download-hf --repo NewBie-AI/NewBie-image-Exp0.1
              uv run python modal_newbie_train.py --lang zh volume-list /jobs
              uv run python modal_newbie_train.py --lang zh volume-download /jobs/<job>/output/<output-name> outputs/<job>/<output-name>
              uv run python modal_newbie_train.py --lang zh job-download --job my-style --config configs/example_lora.toml
            """
            if zh
            else """
            Examples:
              uv run python modal_newbie_train.py train --config configs/example_lokr.toml --dataset D:\\datasets\\my-style --job my-style
              uv run python modal_newbie_train.py train --config configs/example_lora.toml --job resume-my-style --no-upload
              uv run python modal_newbie_train.py train --config configs/example_lora.toml --job long-run --no-upload --detach
              uv run python modal_newbie_train.py model-download-hf --repo NewBie-AI/NewBie-image-Exp0.1
              uv run python modal_newbie_train.py volume-list /jobs
              uv run python modal_newbie_train.py volume-download /jobs/<job>/output/<output-name> outputs/<job>/<output-name>
              uv run python modal_newbie_train.py job-download --job my-style --config configs/example_lora.toml
            """
        ),
    )
    parser.add_argument("--lang", choices=["en", "zh"], default=lang, help="CLI help/output language. Use zh for Chinese.")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="上传输入并运行远程 Modal 训练任务。" if zh else "Upload inputs and run a remote Modal training job.")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--dataset", type=Path)
    train.add_argument("--job", default="")
    train.add_argument("--gpu", default="L40S")
    train.add_argument("--timeout-minutes", type=int, default=360)
    train.add_argument("--volume", default=DEFAULT_VOLUME)
    train.add_argument("--repo-url", default=UPSTREAM_REPO)
    train.add_argument("--trainer-ref", default="main")
    train.add_argument("--no-upload", action="store_true", help="复用 Modal Volume 中已有的配置/数据集。" if zh else "Reuse config/dataset already in the Modal Volume.")
    train.add_argument("--no-install", action="store_true", help="跳过 pip install -r NewbieLoraTrainer/requirements.txt。" if zh else "Skip pip install -r NewbieLoraTrainer/requirements.txt.")
    train.add_argument("--detach", action="store_true", help="提交训练后在本地断开时继续运行。" if zh else "Submit training and keep it running after local disconnect.")
    train.add_argument("--max-return-mb", type=int, default=128)

    volume_list = sub.add_parser("volume-list", help="列出 Modal Volume 中的文件。" if zh else "List files in the Modal Volume.")
    volume_list.add_argument("path", nargs="?", default="/")
    volume_list.add_argument("--volume", default=DEFAULT_VOLUME)

    volume_rm = sub.add_parser("volume-rm", help="从 Modal Volume 删除路径。" if zh else "Remove a path from the Modal Volume.")
    volume_rm.add_argument("path")
    volume_rm.add_argument("--volume", default=DEFAULT_VOLUME)
    volume_rm.add_argument("--yes", action="store_true")

    volume_download = sub.add_parser("volume-download", help="从 Modal Volume 下载文件或目录。" if zh else "Download a file or directory from the Modal Volume.")
    volume_download.add_argument("remote_path")
    volume_download.add_argument("local_path", type=Path)
    volume_download.add_argument("--volume", default=DEFAULT_VOLUME)

    job_download = sub.add_parser("job-download", help="下载训练任务最终 adapter 输出。" if zh else "Download the final adapter output for a training job.")
    job_download.add_argument("--job", required=True, help="训练时使用的 Modal 任务名称。" if zh else "Modal job name used for training.")
    job_download.add_argument("--config", required=True, type=Path, help="用于读取 Model.output_name 的训练 TOML。" if zh else "Training TOML used to get Model.output_name.")
    job_download.add_argument("--output-name", default=None, help="覆盖配置中的 Model.output_name。" if zh else "Override Model.output_name from the config.")
    job_download.add_argument("--local-path", type=Path, default=None, help="本地目标目录。" if zh else "Local destination directory.")
    job_download.add_argument("--volume", default=DEFAULT_VOLUME)

    model_download = sub.add_parser("model-download-hf", help="将 Hugging Face 模型 snapshot 下载到 /workspace/Models。" if zh else "Download a Hugging Face model snapshot into /workspace/Models.")
    model_download.add_argument("--repo", default=DEFAULT_HF_REPO, help="Hugging Face 仓库 ID 或 https://huggingface.co/owner/name URL。" if zh else "Hugging Face repo ID or https://huggingface.co/owner/name URL.")
    model_download.add_argument("--revision", default=None)
    model_download.add_argument("--volume", default=DEFAULT_VOLUME)
    model_download.add_argument("--timeout-minutes", type=int, default=360)
    model_download.add_argument("--hf-secret", default=None, help="包含 HF_TOKEN key 的 Modal Secret 名称。" if zh else "Modal Secret name containing an HF_TOKEN key.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lang = normalize_lang(getattr(args, "lang", "en"))
    save_preferences({"ui_language": lang})
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
            raise SystemExit("拒绝删除：必须传入 --yes。" if lang == "zh" else "Refusing to delete without --yes.")
        remove_volume_path(args.volume, args.path)
        print(f"已从 {args.volume} 删除 {args.path}" if lang == "zh" else f"Removed {args.path} from {args.volume}")
        return

    if args.command == "volume-download":
        bytes_written = download_volume_path(args.volume, args.remote_path, args.local_path)
        print(f"已从 {args.volume} 下载 {args.remote_path} 到 {args.local_path}（{bytes_written} bytes）" if lang == "zh" else f"Downloaded {args.remote_path} from {args.volume} to {args.local_path} ({bytes_written} bytes)")
        return

    if args.command == "job-download":
        result = download_job_output(args.volume, args.job, args.config, args.local_path, args.output_name)
        print(f"已从 {args.volume} 下载 {result['remote_path']} 到 {result['local_path']}（{result['bytes']} bytes）" if lang == "zh" else f"Downloaded {result['remote_path']} from {args.volume} to {result['local_path']} ({result['bytes']} bytes)")
        return

    if args.command == "model-download-hf":
        hf_secret = args.hf_secret if args.hf_secret is not None else (configured_hf_secret_name() or "")
        result = download_hf_model_to_volume(args.repo, args.revision, args.volume, args.timeout_minutes, hf_secret)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result.get("ok") else 1)
