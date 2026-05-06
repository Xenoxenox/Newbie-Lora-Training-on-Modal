# Newbie-Lora-Training-on-Modal

Run LoRA/LoKr training for the Newbie-image model on Modal.

这个项目提供两个入口：

- `modal_newbie_train.py`：无头 CLI，用 Modal 远程运行 Newbie-image LoRA/LoKr 训练。
- `manage.py`：基于 `questionary` 的 TUI，用来生成配置、上传数据集、启动训练和管理 Modal Volume。

实现参考了三个来源：

- `F:\Modal\Faster-Whisper-TransWithAI-ChickenRice\`：轻量本地 submitter、Modal Volume 上传、远端 clone/update、dict payload 返回。
- `F:\Modal-ComfyUI\modal-comfyui\`：Modal Volume 管理和 TUI 菜单风格。
- `https://cnb.cool/xChenNing/Newbie-Lora-Trainer-Public.git`：Newbie 训练入口和配置布局。

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

需要先把 Newbie-image 模型文件放到 Modal Volume 的 `/Models`。可以用 Modal CLI、单独脚本或临时 notebook 上传；训练配置里的 `base_model_path` 默认就是 `/workspace/Models`。

## 无头运行

```powershell
python modal_newbie_train.py train `
  --config configs/example_lokr.toml `
  --dataset D:\datasets\my-style `
  --job my-style `
  --gpu L40S `
  --timeout-minutes 360
```

使用修改版训练入口：

```powershell
python modal_newbie_train.py train --config configs/example_lokr.toml --dataset D:\datasets\my-style --job my-style --xcn
```

续训时可以复用已经上传到 Volume 的配置和数据：

```powershell
python modal_newbie_train.py train --config configs/example_lokr.toml --job my-style --no-upload
```

## TUI

```powershell
python manage.py
```

TUI 支持：

- 生成 LoRA/LoKr job TOML。
- 上传本地数据集并启动 Modal 训练。
- 列出 Modal Volume 路径。
- 确认后删除指定 Volume 路径。

## 配置要点

Newbie 官方训练入口是：

```bash
python NewbieLoraTrainer/train_newbie_lora.py --config_file ./lokr.toml
```

本项目在 Modal 里等价执行：

```bash
python /workspace/Newbie-Lora-Trainer-Public/NewbieLoraTrainer/train_newbie_lora.py --config_file /workspace/jobs/<job>/config.toml
```

`--xcn` 会切换到：

```bash
train_newbie_lora_xcn.py
```

输出较小时，程序会把 `/workspace/jobs/<job>/output` 打包回传到本地 `outputs/<job>/output.zip`。如果输出超过 `--max-return-mb`，产物会留在 Modal Volume 中，路径会显示在运行结果里。
