from __future__ import annotations

from typing import Any


def modal_train(remote_payload: dict[str, Any]) -> dict[str, Any]:
    """Runs inside Modal using the image's Python, not the local launcher Python."""

    import base64
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    import time
    import zipfile

    import modal

    volume = modal.Volume.from_name(remote_payload["volume_name"], create_if_missing=True)
    volume.reload()

    repo_dir = Path(remote_payload["repo_dir"])
    job_dir = Path(remote_payload["remote_job_dir"])
    log_path = Path(remote_payload["remote_log"])
    output_dir = Path(remote_payload["remote_output"])
    tensorboard_dir = Path(remote_payload["remote_tensorboard_dir"])
    config_file = Path(remote_payload["remote_config"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n$ {' '.join(cmd)}\n")
            log.flush()
            proc = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            log.write(proc.stdout)
            log.flush()
        if check and proc.returncode != 0:
            raise RuntimeError(f"Command failed with exit {proc.returncode}: {' '.join(cmd)}")
        return proc

    try:
        if repo_dir.exists() and (repo_dir / ".git").exists():
            run(["git", "fetch", "--depth", "1", "origin", remote_payload["trainer_ref"]], cwd=repo_dir, check=False)
            run(["git", "reset", "--hard", f"origin/{remote_payload['trainer_ref']}"], cwd=repo_dir, check=False)
        else:
            if repo_dir.exists():
                shutil.rmtree(repo_dir)
            run(["git", "clone", "--depth", "1", "--branch", remote_payload["trainer_ref"], remote_payload["repo_url"], str(repo_dir)])

        model_py = repo_dir / "NewbieLoraTrainer" / "models" / "model.py"
        model_text = model_py.read_text(encoding="utf-8")
        old_import = (
            "from flash_attn import flash_attn_varlen_func\n"
            "from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa\n"
        )
        new_import = (
            "try:\n"
            "    from flash_attn import flash_attn_varlen_func\n"
            "    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa\n"
            "except Exception as exc:\n"
            "    print(f\"flash_attn unavailable, falling back to native PyTorch attention: {exc}\")\n"
            "    flash_attn_varlen_func = None\n"
            "    index_first_axis = pad_input = unpad_input = None\n"
        )
        old_condition = "if dtype in [torch.float16, torch.bfloat16] and attn_bias is None:"
        new_condition = "if flash_attn_varlen_func is not None and dtype in [torch.float16, torch.bfloat16] and attn_bias is None:"
        if old_import in model_text:
            model_text = model_text.replace(old_import, new_import)
        model_text = model_text.replace(old_condition, new_condition, 1)
        model_py.write_text(model_text, encoding="utf-8")

        trainer_py = repo_dir / "NewbieLoraTrainer" / "train_newbie_lora.py"
        trainer_text = trainer_py.read_text(encoding="utf-8")
        dtype_block = (
            "    if mixed_precision == 'bf16':\n"
            "        model_dtype = torch.bfloat16\n"
            "    elif mixed_precision == 'fp16':\n"
            "        model_dtype = torch.float16\n"
            "    else:\n"
            "        model_dtype = torch.float32\n"
        )
        patched_dtype_block = (
            "    if mixed_precision == 'bf16':\n"
            "        model_dtype = torch.bfloat16\n"
            "        clip_torch_dtype = \"bfloat16\"\n"
            "    elif mixed_precision == 'fp16':\n"
            "        model_dtype = torch.float16\n"
            "        clip_torch_dtype = \"float16\"\n"
            "    else:\n"
            "        model_dtype = torch.float32\n"
            "        clip_torch_dtype = \"float32\"\n"
        )
        if dtype_block not in trainer_text:
            raise RuntimeError(f"Unable to patch dtype block in {trainer_py}")
        trainer_text = trainer_text.replace(dtype_block, patched_dtype_block)
        clip_replacements = {
            "clip_model = AutoModel.from_pretrained(clip_model_path, torch_dtype=model_dtype, trust_remote_code=True)":
                "clip_model = AutoModel.from_pretrained(clip_model_path, torch_dtype=clip_torch_dtype, trust_remote_code=True)",
            "clip_model = AutoModel.from_pretrained(clip_path, torch_dtype=model_dtype, trust_remote_code=True)":
                "clip_model = AutoModel.from_pretrained(clip_path, torch_dtype=clip_torch_dtype, trust_remote_code=True)",
        }
        for old_call, new_call in clip_replacements.items():
            if old_call not in trainer_text:
                raise RuntimeError(f"Unable to patch CLIP dtype call in {trainer_py}")
            trainer_text = trainer_text.replace(old_call, new_call)
        tokenizer_replacements = {
            "clip_tokenizer = AutoTokenizer.from_pretrained(clip_model_path, trust_remote_code=True)":
                "clip_tokenizer = AutoTokenizer.from_pretrained(clip_model_path, trust_remote_code=True, fix_mistral_regex=False)",
            "clip_tokenizer = AutoTokenizer.from_pretrained(clip_path, trust_remote_code=True)":
                "clip_tokenizer = AutoTokenizer.from_pretrained(clip_path, trust_remote_code=True, fix_mistral_regex=False)",
        }
        for old_call, new_call in tokenizer_replacements.items():
            if old_call not in trainer_text:
                raise RuntimeError(f"Unable to patch CLIP tokenizer call in {trainer_py}")
            trainer_text = trainer_text.replace(old_call, new_call)
        tracker_project_old = "    output_dir = config['Model']['output_dir']\n    os.makedirs(output_dir, exist_ok=True)\n"
        tracker_project_new = (
            "    output_dir = config['Model']['output_dir']\n"
            "    os.makedirs(output_dir, exist_ok=True)\n"
            "    logging_dir = config['Model'].get('logging_dir') or output_dir\n"
            "    os.makedirs(logging_dir, exist_ok=True)\n"
        )
        if tracker_project_old not in trainer_text:
            raise RuntimeError(f"Unable to patch TensorBoard logging dir setup in {trainer_py}")
        trainer_text = trainer_text.replace(tracker_project_old, tracker_project_new, 1)
        tracker_project_arg = "        project_dir=output_dir,"
        if tracker_project_arg not in trainer_text:
            raise RuntimeError(f"Unable to patch TensorBoard project dir in {trainer_py}")
        trainer_text = trainer_text.replace(tracker_project_arg, "        project_dir=logging_dir,", 1)
        trainer_py.write_text(trainer_text, encoding="utf-8")

        requirements = repo_dir / "NewbieLoraTrainer" / "requirements.txt"
        requirements_text = requirements.read_text(encoding="utf-8")
        requirements_text = requirements_text.replace("transformers>=4.38.0", "transformers>=4.38.0,<5")
        if "setuptools" not in requirements_text.lower():
            requirements_text = requirements_text.rstrip() + "\nsetuptools<81\n"
        if "tensorboard" not in requirements_text.lower():
            requirements_text = requirements_text.rstrip() + "\ntensorboard>=2.16.0\n"
        requirements.write_text(requirements_text, encoding="utf-8")
        if remote_payload["install_requirements"]:
            run([sys.executable, "-m", "pip", "install", "-U", "-r", str(requirements)])

        import toml

        trainer_config = toml.load(str(config_file))
        trainer_model_config = trainer_config.setdefault("Model", {})
        trainer_model_config["logging_dir"] = str(tensorboard_dir)
        config_file.write_text(toml.dumps(trainer_config), encoding="utf-8")

        command = [
            sys.executable,
            str(trainer_py),
            "--config_file",
            str(config_file),
        ]

        started = time.time()
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n$ {' '.join(command)}\n")
            log.flush()
            proc = subprocess.Popen(
                command,
                cwd=str(repo_dir),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
            assert proc.stdout is not None
            stream_training_output = True
            for line in proc.stdout:
                if stream_training_output:
                    try:
                        print(line, end="")
                    except BrokenPipeError:
                        stream_training_output = False
                log.write(line)
                log.flush()
            return_code = proc.wait()
            if return_code == 0:
                tail_notice = (
                    "\nTraining process exited successfully. The Modal app will close automatically.\n"
                    "If a BrokenPipeError appears after 'Training finished', it is harmless stdout shutdown noise.\n"
                )
                log.write(tail_notice)
                log.flush()
                if stream_training_output:
                    try:
                        print(tail_notice, end="")
                    except BrokenPipeError:
                        pass

        zip_path = job_dir / "output.zip"
        artifacts = []
        if output_dir.exists():
            for path in output_dir.rglob("*"):
                if path.is_file():
                    artifacts.append(
                        {
                            "path": str(path),
                            "bytes": path.stat().st_size,
                            "relative": str(path.relative_to(output_dir)),
                        }
                    )
            if artifacts:
                with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for item in artifacts:
                        archive.write(item["path"], item["relative"])

        returned_zip = None
        max_bytes = int(remote_payload["max_return_mb"]) * 1024 * 1024
        if zip_path.exists() and zip_path.stat().st_size <= max_bytes:
            returned_zip = {
                "name": zip_path.name,
                "bytes": zip_path.stat().st_size,
                "base64": base64.b64encode(zip_path.read_bytes()).decode("ascii"),
            }

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        result = {
            "ok": return_code == 0,
            "return_code": return_code,
            "seconds": round(time.time() - started, 2),
            "job_dir": str(job_dir),
            "output_dir": str(output_dir),
            "tensorboard_volume_path": str(remote_payload["volume_tensorboard_dir"]),
            "tensorboard_dir": str(tensorboard_dir),
            "log_path": str(log_path),
            "artifacts": artifacts,
            "returned_zip": returned_zip,
            "log_tail": log_text[-12000:],
        }
    except Exception as exc:
        if log_path.exists():
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-12000:]
        else:
            log_tail = ""
        result = {
            "ok": False,
            "return_code": 999,
            "error": repr(exc),
            "job_dir": str(job_dir),
            "output_dir": str(output_dir),
            "tensorboard_volume_path": str(remote_payload["volume_tensorboard_dir"]),
            "tensorboard_dir": str(tensorboard_dir),
            "log_path": str(log_path),
            "artifacts": [],
            "returned_zip": None,
            "log_tail": log_tail,
        }

    volume.commit()
    return result
