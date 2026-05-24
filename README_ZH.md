# Newbie LoRA Training on Modal

在本地终端中运行 NewBie-image LoRA/LoKr 训练。

English version: [README.md](README.md)

本项目把本地机器作为控制端，通过小型 TUI 或 headless CLI 完成Newbie-image底模同步、数据集上传、训练、结果下载和 Volume 管理。GPU 计算发生在 Modal 容器中，训练输入和输出保存在 Modal Volume 里。

## 适用场景

本项目适用于以下场景：

- 无需维护GPU服务器即可训练 NewBie-image LoRA 或 LoKr，无存储费用焦虑，笔记本也可随时开炉；
- 把 NewBie 底模和任务产物保存在持久化的 Modal Volume 中，远程容器直接读取底模快照，无需重复部署底模；
- 通过引导式提示创建训练配置TOML，无需手工编辑每个字段；
- 本地数据集上传到可复用的 `/jobs/<job>/dataset` 路径，支持重复训练；
- 用户可以选择带实时日志的 attached 训练，或在断开本地终端后继续运行的 detached 训练；detached 训练可以有效避免本地终端断连导致的训练中断。
- 把训练产物一键下载回本地 `outputs/`目录。

Modal 按使用量计费。每人每月享30美元使用额度以及1TB的免费Volume容量。查看最新GPU价格表：

https://modal.com/pricing

## 前置条件

- 一个 Modal 账号（需要信用卡做KYC验证，支持银联）
- 本地安装 Python 3.11+
- Hugging Face token(可选)；HF token可以让Modal加速下载模型文件并访问私人仓库，减少NewBie底模初次部署时间

使用一键脚本安装本地依赖：

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

Linux 或 macOS：

```bash
bash ./setup.sh
```

脚本会检测操作系统，超时时自动切换到清华 PyPI 镜像，并让 `uv sync` 创建或更新 `.venv`。

如果你想手动执行：

```powershell
uv sync
uv run modal setup
```

项目命令请使用 `uv run ...`。

如果你不想每次都输入，请进入项目虚拟环境。

Windows:

```powershell
.\.venv\Scripts\activate
```

Linux 或 macOS：

```bash
source ./.venv/Scripts/activate
```

安装脚本不会自动执行 Modal 登录；初次使用时，请运行 `uv run modal setup`。TUI 在启动时也会检测是否缺少 Modal token，并能直接帮你启动 `modal setup`。

## 标准训练流程：

在训练开始前，请确保你已经准备好本地训练集了。

0. 配置 Modal Secret（该步不是必须的，但十分推荐，因为它会加快底模的初次部署）；
1. 加载模型（底模）到Volume，首次执行成功后，后续训练流程可以跳过此步；
2. 创建训练配置（ LoRA/LoKr TOML）；
3. 启动训练；
4. 下载任务输出（训练产物）。

以下是使用本项目TUI执行标准训练流程的详细教程。

### TUI：一、运行与语言选择

启动引导式训练向导：

```powershell
uv run python manage.py
```

启动时选择 `English` 或 `中文`。语言偏好会保存到 `.modal-newbie/preferences.json`，后续启动时会复用。需要切换语言时，在启动语言提示里重新选择即可。

### TUI：二、同步底模

在第一次训练前，先在 TUI 中选择 `加载模型到 Volume`。

`超时时间（分钟）`填默认值。

`包含 HF_TOKEN 的 Modal Secret 名称`  填默认值。

TUI 始终使用这个底模仓库：

```text
NewBie-AI/NewBie-image-Exp0.1
```

它会把NewBie底模快照下载到：

```text
/workspace/Models
```


### TUI：三、创建任务配置

在 TUI 中选择 `创建配置`。

根据文本指引和你的实际需求填写配置。

重要提示项：

- `配置/任务名称`：用于 Modal 路径和配置文件名的任务 slug；
- `Adapter 类型`： `LoKr` 或 `LoRA`；
- `输出文件夹名称`： `下载任务输出` 时，TUI会用它定位下载目录；
- `训练分辨率`：官方示例推荐 `1024`；更高分辨率会消耗更多显存；
- `训练轮数`：即Epochs，官方示例中 LoKr 用 `24`，LoRA 用 `30`；
- `批大小`：即Batch Size，官方示例使用 `4`；TUI 默认 `1`，对高分辨率 Modal 运行更稳妥；
- `学习率`：即Learning rate，官方示例中 LoKr 使用 `3e-4`，LoRA 使用 `1e-4`。

生成的配置是本地文件，不包含 Hugging Face token。

### TUI：四、运行训练

在 TUI 中选择 `Run Training`。

向导会询问以下内容：

- 训练配置；
- Modal 任务名称；
- 数据集来源；
- GPU 类型；
- 超时时间；
- 启动模式。

启动模式：

- `Attached (滚动日志模式)`：训练结束前在本地持续流式输出 Modal 日志；
- `Detached (后台运行模式)`：提交任务后返回控制权，让 Modal 在后台继续运行。

提交前，TUI 会显示一份预训练摘要，包含任务名、config、数据集、上传模式、GPU、超时时间、预计最高 GPU 成本、启动模式和依赖。

提交后，Modal 可能会先分配 GPU、检查或构建容器镜像，然后训练日志才会出现。TUI 会先打印明确的资源分配/镜像状态，再在 Modal 返回 App ID 后显示 App ID、dashboard 链接、function call 和可复制的 `modal app logs ... --follow` 命令。

运行时依赖安装路径仍可通过 CLI 作为高级兜底方案；TUI 主路径不会提示它。
baked image 也会安装匹配 CUDA/PyTorch 运行环境的预编译 FlashAttention wheel，因此 `use_flash_attention_2 = true` 的配置不需要在运行时编译 FlashAttention。

TensorBoard 标量日志默认启用。runner 会把 event 文件写到：

```text
/workspace/jobs/<job>/output/tensorboard
```

任务完成后，结果面板会包含远程输出路径、TensorBoard 路径、训练日志路径、本地 app log、App ID、dashboard 链接、日志命令，以及 `modal app stop <app-id>` 命令。Modal 通常会自动关闭 app；只有在 app 看起来滞留时才使用 stop 命令。

> [!TIP]
> 训练支持断点续传。
> 要想续训中断的任务，只需要选择和中断任务相同的 `训练配置`和 `Modal 任务名称`，再次执行训练任务。

## TUI：五、下载训练产物

在训练任务完成后返回TUI主菜单，选择 `下载任务输出`。

配置选择器会读取训练配置文件，结合 Modal 任务名，它会解析远程目录：

```text
/jobs/<job>/output/<output_name>
```

默认情况下，训练产物会下载到以下本地目录：

```text
./outputs/<job>/<output_name>
```

训练产物包含：
- `adapter_model.safetensors`：LoRA模型文件；
- `adapter_config.json`：adapter配置文件；
- `README.md`：基础说明文件。

### TUI：其他选项的说明

#### 配置 Modal Secret

创建一个包含 `HF_TOKEN` key 的 Modal Secret。

默认 Secret 名称是 `LoRATraining`；在 `Modal Account` 检查成功后，TUI 会在 `Modal Secrets` 面板里显示它。你也可以通过 `Configure Modal Secrets` 创建或更新它。真实 token 只应输入到密码框或 Modal CLI，不要写进文档、配置、命令面板或日志。

如果当前 Modal Environment 中找不到配置的 Secret，模型同步会打印警告并在不注入 Secret 的情况下继续运行。这样可以避免底模下载在任务开始前就失败。Modal 的认证或网络错误仍会正常暴露。

你可以用Modal CLI来查看或管理Modal Secret：

```bash
uv run modal secret -h
```

你也可以把非敏感的 Secret 名称持久化到 gitignored 的 `config.toml` 中，供后续 TUI 运行和手动 CLI 命令使用：

```toml
[modal.secrets]
hf_secret_name = "YourSecretName"
```

本地环境变量会覆盖 `config.toml`，而 `config.toml` 会覆盖默认的 `LoRATraining` 名称。将 `MODAL_HF_SECRET_NAME` 或 `hf_secret_name` 设为空字符串、 `none` 或 `false` 会禁用 Secret 注入。不要把 token 值写入 TOML、README 示例、命令面板或日志。

#### 管理Volume

在 TUI 中选择 `管理Volume` 可以：

- 列出 Modal Volumes；
- 删除 `/jobs/<job>` 下的任务目录；
- 删除整个 Volume；
- 重命名 Volume；
- 打开 Volume dashboard。

有破坏性的 TUI 操作都需要确认。CLI 也要求通过 `--yes` 才能删除 Volume。

默认 Volume 名称是：

```text
newbie-image-lora
```

#### 状态栏与退出快捷键

- `Modal Account`：本地 Modal token/profile 是否可用；
- `Modal Secrets`：已配置的 Hugging Face Secret 是否存在。

这两个检查是分开的。如果 Modal 账号 token 缺失，Secret 检查会跳过，直到登录成功。出现提示时选择 Yes 运行 `modal setup`；浏览器流程结束后，TUI 会自动刷新账号和 Secret 状态，无需重启。

在子菜单中，`Ctrl+C` 返回主菜单。在主菜单中，`Ctrl+C` 退出。

退出时，会根据 TUI 的启动时间和退出时间，显示会话账单摘要。

## Headless CLI

正常使用推荐 TUI。CLI 适合自动化和高级工作流。

使用 `--lang zh` 可以查看中文 CLI 帮助和中文状态提示：

```powershell
uv run python modal_newbie_train.py --lang zh --help
uv run python modal_newbie_train.py --lang zh train --help
```

同步默认底模：

```powershell
uv run python modal_newbie_train.py model-download-hf
```

带上传训练：

```powershell
uv run python modal_newbie_train.py train `
  --config configs/example_lokr.toml `
  --dataset D:\datasets\my-style `
  --job my-style `
  --gpu L40S `
  --timeout-minutes 360
```

复用已上传的任务配置和数据集：

```powershell
uv run python modal_newbie_train.py train `
  --config configs/example_lokr.toml `
  --job my-style `
  --no-upload
```

提交 detached 后台运行：

```powershell
uv run python modal_newbie_train.py train `
  --config configs/example_lora.toml `
  --job my-style `
  --no-upload `
  --detach
```

跳过上游 trainer requirements 的运行时依赖安装：

```powershell
uv run python modal_newbie_train.py train `
  --config configs/example_lokr.toml `
  --dataset D:\datasets\my-style `
  --job my-style `
  --no-install
```

根据 config 中的 `Model.output_name` 下载已完成任务输出：

```powershell
uv run python modal_newbie_train.py job-download `
  --job my-style `
  --config configs/example_lokr.toml
```

下载 TensorBoard event 文件并在本地查看：

```powershell
uv run python modal_newbie_train.py volume-download /jobs/my-style/output/tensorboard outputs/my-style/tensorboard
uv run tensorboard --logdir outputs/my-style/tensorboard
```

列出并下载 Volume 路径：

```powershell
uv run python modal_newbie_train.py volume-list /jobs
uv run python modal_newbie_train.py volume-download /jobs/my-style/output/my-style outputs/my-style/my-style
```

删除 Volume 路径：

```powershell
uv run python modal_newbie_train.py volume-rm /jobs/my-style --yes
```

## Modal Workspace 布局

Modal 会把训练 Volume 挂载到 `/workspace`。

```text
/workspace/Models                       # NewBie 底模 snapshot
/workspace/jobs/<job>/config.toml       # 上传的任务配置
/workspace/jobs/<job>/dataset           # 上传的数据集
/workspace/jobs/<job>/output            # adapter 输出
/workspace/jobs/<job>/output/tensorboard # TensorBoard event 文件
/workspace/jobs/<job>/logs/train.log    # 远程 trainer 日志
```

本地仓库存放运行时产物的忽略目录：

```text
logs/       # 本地 Modal app 日志
outputs/    # 下载的结果
configs/jobs/ # 生成的任务配置
.modal-newbie/preferences.json # 非敏感 TUI 偏好
config.toml # 非敏感的本地 Modal Secret 名称
```

偏好文件只保存非敏感内容，例如 UI 语言、上次使用的 config、GPU、timeout 和 run mode，不会保存 token。

## Trainer Source

远程训练参考上游 Newbie trainer：

https://cnb.cool/xChenNing/Newbie-Lora-Trainer-Public

在 Modal 内，这个项目运行的是：

```bash
python /workspace/Newbie-Lora-Trainer-Public/NewbieLoraTrainer/train_newbie_lora.py --config_file /workspace/jobs/<job>/config.toml
```

Modal 训练镜像会预先打包稳定的训练依赖。镜像会先安装 CUDA 12.4 PyTorch wheel，再安装匹配的预编译 `flash-attn==2.7.4.post1` wheel，最后安装其他 trainer 依赖。运行时的 `install_requirements` 逻辑仍保留在 headless runner 中作为高级兜底，但 TUI 主路径默认使用 baked image。

上游 trainer 补丁仍保留 FlashAttention 可选策略：如果 `flash_attn` 无法导入，训练会 fallback 到原生 PyTorch attention 路径，而不是在导入阶段失败。

## 安全说明

- 不要提交 `.env`、数据集、模型文件、日志、输出或下载的参考仓库；
- 保护好 Modal 凭据和 Volume 内容；
- Hugging Face token 请放进 Modal Secrets，不要写进 TOML、README 示例、命令面板或日志；
- Modal token ID 和 token secret 只放在 Modal 本地配置或环境变量中，绝不要贴到 issue、文档、提交或日志里；
- `config.toml` 可以在 `[modal.secrets]` 下存放 Modal Secret 名称，但不要存 token 值；
- 上传任务前请确认任务名：上传会删除并替换该任务旧的 `/config.toml` 和 `/dataset`；
- 注意 Volume 名称：部分操作可以通过名称创建或定位 Modal Volume。

## 命令参考

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
uv sync
uv run modal setup
uv run python manage.py
uv run python modal_newbie_train.py model-download-hf
uv run python modal_newbie_train.py train --config configs/example_lokr.toml --dataset D:\datasets\my-style --job my-style --gpu L40S
uv run python modal_newbie_train.py train --config configs/example_lokr.toml --job my-style --no-upload --detach
uv run python modal_newbie_train.py job-download --job my-style --config configs/example_lokr.toml
uv run python modal_newbie_train.py volume-list /jobs
```

## 贡献

请把改动聚焦在用户工作流上。对于 Python 改动，请运行：

```powershell
uv run python -m py_compile modal_newbie_train.py manage.py scripts/tui.py scripts/config_flow.py scripts/volume_flow.py scripts/training_flow.py scripts/billing.py scripts/training_core.py scripts/model_ops.py scripts/volume_ops.py scripts/cli.py scripts/preferences.py scripts/secret_config.py
git diff --check
```

对于 Modal 训练相关改动，请记录 config、GPU、是否上传，以及运行模式是 attached 还是 detached。
