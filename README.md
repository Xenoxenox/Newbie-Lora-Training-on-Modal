# Newbie-Lora-Training-on-Modal

Run LoRA/LoKr training for the Newbie-image model on Modal.

这个项目提供两个入口：

- `modal_newbie_train.py`：无头 CLI，用 Modal 远程运行 Newbie-image LoRA/LoKr 训练。
- `manage.py`：基于 `questionary` 的 TUI，用来生成配置、上传数据集、启动训练和管理 Modal Volume。

实现参考了三个来源：

- [TransWithAI/Faster-Whisper-TransWithAI-ChickenRice](https://github.com/TransWithAI/Faster-Whisper-TransWithAI-ChickenRice)：轻量本地 submitter、Modal Volume 上传、远端 clone/update、dict payload 返回。
- [Xenoxenox/modal-comfyui](https://github.com/Xenoxenox/Newbie-Lora-Training-on-Modal)：Modal Volume 管理和 TUI 菜单风格。
- [xChenNing/Newbie-Lora-Trainer-Public](https://cnb.cool/xChenNing/Newbie-Lora-Trainer-Public/-/tree/main)：Newbie 训练入口和配置布局。

## 目录结构

```text
Newbie-Lora-Training-on-Modal/
├── manage.py                     # 交互式 TUI：生成配置、上传数据、启动训练、管理 Volume
├── modal_newbie_train.py         # 无头 CLI：提交 Modal 远程训练任务
├── requirements.txt              # 本地运行依赖
├── README.md                     # 项目说明
├── AGENTS.md                     # 贡献者与代理协作指南
├── configs/
│   ├── example_lora.toml         # LoRA 示例配置
│   ├── example_lokr.toml         # LoKr 示例配置
│   └── jobs/                     # TUI 生成的 job 配置
├── outputs/                      # 小体积训练产物回传目录，Git 忽略
└── logs/                         # 本地 Modal App 日志目录，仅保留 .gitkeep
```

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
modal setup
```

## 训练前准备

Modal Volume 会挂载到远端 `/workspace`。默认布局：

```text
/workspace/Newbie-Lora-Trainer-Public   # 远端自动 clone/update
/workspace/Models                       # Newbie-image 基础模型目录
/workspace/jobs/<job>/dataset           # 上传的数据集
/workspace/jobs/<job>/config.toml       # 上传的训练配置
/workspace/jobs/<job>/output            # 训练输出
```

需要先把 Newbie-image 模型文件放到 Modal Volume 的 `/Models`。训练配置里的 `base_model_path` 默认就是 `/workspace/Models`，本项目通过 TUI 或 CLI 从 Hugging Face 下载 snapshot 到该目录。

如果模型仓库需要 Hugging Face token，先在 Modal 创建 Secret。默认 Secret 名称为 `LoRATraining`，其中的键名为 `HF_TOKEN`：

```powershell
modal secret create LoRATraining HF_TOKEN=hf_xxx
```

从 Hugging Face 下载完整 diffusers snapshot 到 `/workspace/Models`（默认仓库 `NewBie-AI/NewBie-image-Exp0.1`）：

```powershell
# 使用默认仓库
python modal_newbie_train.py model-download-hf

# 指定其他仓库
python modal_newbie_train.py model-download-hf --repo owner/name

# 指定 revision
python modal_newbie_train.py model-download-hf --revision main
```

## 无头运行

```powershell
python modal_newbie_train.py train `
  --config configs/example_lokr.toml `
  --dataset D:\datasets\my-style `
  --job my-style `
  --gpu L40S `
  --timeout-minutes 360
```

续训时可以复用已经上传到 Volume 的配置和数据：

```powershell
python modal_newbie_train.py train --config configs/example_lokr.toml --job my-style --no-upload
```

长时间训练可使用 detached 模式，提交后本地断开也不会取消远程训练：

```powershell
python modal_newbie_train.py train --config configs/example_lora.toml --job my-style --no-upload --detach
```

attached 模式会实时滚动输出 Modal App 日志，并保存到本地 `logs/modal_app_<job>_<timestamp>.log`。detached 模式只提交远端任务，不在本地持续追踪日志；可使用结果里的 App ID 或 Function Call ID 到 Modal Dashboard 查看。

## TUI

```powershell
python manage.py
```

TUI 支持：

- 生成 LoRA/LoKr job TOML。
- 从 Hugging Face 下载基础模型到 Modal Volume。
- 分步骤上传本地数据集并启动 Modal 训练，提交前会显示 pre-flight review，可选择 detached 模式。
- 下载训练完成后的 adapter 输出目录。
- Volume 管理：列出、删除、重命名 Volume，删除指定 `/jobs/<job>` 目录，打开 Dashboard。
- attached 训练结束后在结果面板显示本地 Modal App 日志路径；失败时仍显示远端训练日志 tail。

## 配置要点

Newbie 官方训练入口是：

```bash
python NewbieLoraTrainer/train_newbie_lora.py --config_file ./lokr.toml
```

本项目在 Modal 里等价执行：

```bash
python /workspace/Newbie-Lora-Trainer-Public/NewbieLoraTrainer/train_newbie_lora.py --config_file /workspace/jobs/<job>/config.toml
```

上游 `train_newbie_lora_xcn.py` 已弃用；该脚本逻辑未针对当前 Modal 场景优化，本项目固定使用 `train_newbie_lora.py`。

输出较小时，程序会把 `/workspace/jobs/<job>/output` 打包回传到本地 `outputs/<job>/output.zip`。如果输出超过 `--max-return-mb`，产物会留在 Modal Volume 中，路径会显示在运行结果里。
