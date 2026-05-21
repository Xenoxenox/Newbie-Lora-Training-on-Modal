from __future__ import annotations

from dataclasses import dataclass


LANG_EN = "en"
LANG_ZH = "zh"
SUPPORTED_LANGS = {LANG_EN, LANG_ZH}
DEFAULT_LANG = LANG_EN


@dataclass(frozen=True)
class LanguagePack:
    code: str
    app_title: str
    app_subtitle: str
    main_menu: str
    main_run_training: str
    main_run_training_desc: str
    main_download_output: str
    main_download_output_desc: str
    main_load_model: str
    main_load_model_desc: str
    main_create_config: str
    main_create_config_desc: str
    main_manage_volumes: str
    main_manage_volumes_desc: str
    main_quit: str
    main_quit_desc: str
    language_select_title: str
    language_select_prompt: str
    language_select_desc: str
    language_save_confirm: str
    modal_setup_prompt: str
    modal_setup_start: str
    modal_setup_done: str
    modal_setup_retry: str
    modal_setup_skipped: str
    returned_to_menu: str
    step_config_identity: str
    step_adapter: str
    step_training_defaults: str
    step_launch_options: str
    step_summary: str
    config_job_name: str
    adapter_type: str
    adapter_lokr_desc: str
    adapter_lora_desc: str
    output_model_name: str
    training_resolution: str
    resolution_1024_desc: str
    resolution_768_desc: str
    resolution_1536_desc: str
    epochs: str
    batch_size: str
    learning_rate: str
    training_config: str
    create_new_config: str
    create_new_config_desc: str
    modal_job_name: str
    first_upload_question: str
    first_upload_instruction: str
    dataset_directory: str
    dataset_directory_instruction: str
    gpu: str
    timeout_minutes: str
    install_question: str
    install_instruction: str
    detach_question: str
    detach_instruction: str
    volume_management: str
    volume_list: str
    volume_list_desc: str
    volume_delete: str
    volume_delete_desc: str
    volume_rename: str
    volume_rename_desc: str
    volume_dashboard: str
    volume_dashboard_desc: str
    back: str
    hf_repo: str
    revision: str
    hf_secret: str
    download_job_name: str
    remote_output_override: str
    local_destination: str
    session_closed: str
    log_tail: str
    preflight_check: str
    modal_account: str
    modal_secrets: str
    upload_new: str
    reuse_volume: str
    success: str
    failed: str
    available_folders: str
    no_volumes_found: str
    no_jobs_found: str
    create_new_job: str
    create_new_job_desc: str
    config_examples: str
    replace_existing_config: str
    config_exists_retry: str
    allowed_name_chars: str
    name_preview: str
    use_slug_confirm: str
    use_slug_detail: str
    dataset_folder_hint: str
    balanced_default: str
    lower_memory: str
    higher_detail: str
    official_lokr_default: str
    smaller_lora: str
    images_per_step: str
    learning_rate_hint: str
    volume_delete_warning: str
    volume_rename_title: str
    volume_dashboard_opened: str
    job_directory_deleted: str
    download_failed: str
    base_model: str
    model_target: str
    model_load_finished: str
    model_load_failed: str
    training_job_submitted: str
    remote_training_finished: str
    remote_training_failed: str
    training_submission_cancelled: str
    modal_account_title: str
    modal_secrets_title: str
    modal_secret_configure: str
    modal_secret_back: str
    modal_secret_tokens_note: str
    modal_secret_name: str
    modal_secret_password: str
    modal_secret_saved: str
    modal_secret_unchanged: str
    modal_secret_disabled: str
    modal_secret_missing: str
    modal_secret_skipped: str
    modal_secret_unknown: str
    modal_secret_ok: str
    modal_secret_disabled_via: str
    modal_setup_missing_token: str
    modal_setup_complete: str
    modal_setup_exit_code: str
    modal_setup_browser_note: str
    modal_setup_browser_done: str
    modal_setup_browser_retry: str
    app_running: str
    app_dashboard: str
    function_call_id: str
    function_call_dashboard: str
    live_logs: str
    creating_volume_model: str
    track_progress: str
    detached_submit: str
    detached_continue: str
    remote_allocating: str
    synchronous_warning: str
    volume_name: str
    job_directory_to_delete: str
    current_volume_name: str
    new_volume_name: str
    volume_name_to_delete: str
    config_for_output_lookup: str
    output_name_override_hint: str
    local_destination_hint: str
    no_secret_name: str
    no_token: str
    volume_list_title: str
    session_billing_unavailable: str
    session_billing_note: str
    session_billing_wait: str
    session_total_cost: str
    session_rows: str
    modal_profile: str
    billing_window: str
    dashboard: str
    session_start: str
    session_end: str
    training_live_logs: str
    sync_base_model: str


EN = LanguagePack(
    code=LANG_EN,
    app_title="NewBie-image LoRA Training Wizard",
    app_subtitle="Guided Modal workflow for NewBie LoRA/LoKr jobs",
    main_menu="What do you want to do?",
    main_run_training="Run Training",
    main_run_training_desc="Start a NewBie LoRA/LoKr training job.",
    main_download_output="Download Results",
    main_download_output_desc="Fetch a completed adapter folder from the Modal Volume.",
    main_load_model="Sync Base Model",
    main_load_model_desc="Download the NewBie base model snapshot into /workspace/Models.",
    main_create_config="Create Job Config",
    main_create_config_desc="Generate a LoRA or LoKr TOML config for a new job.",
    main_manage_volumes="Volume Maintenance",
    main_manage_volumes_desc="List, rename, delete, or open Modal Volumes.",
    main_quit="Quit",
    main_quit_desc="Exit without changing anything else.",
    language_select_title="Language",
    language_select_prompt="Choose the UI language",
    language_select_desc="This preference is stored locally and reused next time.",
    language_save_confirm="Remember this language for next time?",
    modal_setup_prompt="Modal token is missing. Do you want to run 'modal setup' now?",
    modal_setup_start="Starting Modal setup. Complete the browser flow, then return here.",
    modal_setup_done="Modal setup finished. Refreshed status is shown below.",
    modal_setup_retry="Modal setup exited with code {code}. You can retry from the terminal or continue in the TUI.",
    modal_setup_skipped="Modal setup skipped.",
    returned_to_menu="Returned to main menu.",
    step_config_identity="Step 1: Config Identity",
    step_adapter="Step 2: Adapter",
    step_training_defaults="Step 3: Training Defaults",
    step_launch_options="Step 4: Launch Options",
    step_summary="Step 5: Summary",
    config_job_name="Config/job name",
    adapter_type="Adapter type",
    adapter_lokr_desc="Recommended default for Newbie-image training quality.",
    adapter_lora_desc="Smaller adapter for quicker, lower-memory experiments.",
    output_model_name="Result folder name",
    training_resolution="Training resolution",
    resolution_1024_desc="Balanced default.",
    resolution_768_desc="Lower memory and faster iterations.",
    resolution_1536_desc="Higher detail with higher GPU cost.",
    epochs="Epochs",
    batch_size="Batch size",
    learning_rate="Learning rate",
    training_config="Training config",
    create_new_config="Create a new config",
    create_new_config_desc="Build a fresh LoRA/LoKr job config through guided prompts.",
    modal_job_name="Modal job name",
    first_upload_question="Is this dataset being uploaded to this job for the first time?",
    first_upload_instruction="Choose No only when this job's config and dataset already exist in Modal Volume.",
    dataset_directory="Local dataset directory",
    dataset_directory_instruction="Folder containing images and captions. e.g., ./data/my_images or D:\\datasets\\style",
    gpu="GPU",
    timeout_minutes="Timeout minutes",
    install_question="Install or update trainer dependencies before training?",
    install_instruction="Choose Yes for the first run or after trainer updates. Choose No only when the remote environment is already prepared.",
    detach_question="Keep training running after the local terminal disconnects?",
    detach_instruction="Detached mode submits the job and lets Modal keep running. The local terminal returns control.",
    volume_management="Volume management",
    volume_list="List volumes",
    volume_list_desc="Show available Modal Volumes in this account.",
    volume_delete="Delete a volume",
    volume_delete_desc="Permanently remove a Volume and all data inside it.",
    volume_rename="Rename a volume",
    volume_rename_desc="Change a Volume name without downloading its contents.",
    volume_dashboard="Open dashboard",
    volume_dashboard_desc="Open the selected Volume in the Modal dashboard.",
    back="Back",
    hf_repo="Hugging Face repo or URL",
    revision="Revision",
    hf_secret="Modal Secret name containing HF_TOKEN",
    download_job_name="Modal job name used for training",
    remote_output_override="Remote output folder override",
    local_destination="Local destination",
    session_closed="Session Closed",
    log_tail="Log Tail",
    preflight_check="PRE-FLIGHT CHECK",
    modal_account="Modal Account",
    modal_secrets="Modal Secrets",
    upload_new="UPLOAD NEW",
    reuse_volume="REUSE VOLUME",
    success="SUCCESS",
    failed="FAILED",
    available_folders="Available folders",
    no_volumes_found="No volumes found.",
    no_jobs_found="No job directories found in /jobs.",
    create_new_job="Create a new job config",
    create_new_job_desc="Build a fresh LoRA/LoKr job config through guided prompts.",
    config_examples="Example or hand-written config.",
    replace_existing_config="Replace existing config '{path}'?",
    config_exists_retry="Choose No to enter a different config name.",
    allowed_name_chars="Allowed: letters, numbers, dots, underscores, hyphens.",
    name_preview="Name preview:",
    use_slug_confirm="Use '{slug}' as the {noun} slug?",
    use_slug_detail="Your input will be stored as '{slug}' for Modal paths, config files, and local folders.",
    dataset_folder_hint="Folder containing images and captions. e.g., ./data/my_images or D:\\datasets\\style",
    balanced_default="Balanced default.",
    lower_memory="Lower memory and faster iterations.",
    higher_detail="Higher detail with higher GPU cost.",
    official_lokr_default="Recommended default for Newbie-image training quality.",
    smaller_lora="Smaller adapter for quicker, lower-memory experiments.",
    images_per_step="Images per GPU step. Official examples use 4; keep 1 for high resolutions or limited GPU memory.",
    learning_rate_hint="Optimizer step size. Official examples: {official}; upstream comments recommend testing 1e-4 or 2e-4.",
    volume_delete_warning="This cannot be undone from the TUI.",
    volume_rename_title="Volume Renamed",
    volume_dashboard_opened="Dashboard Opened",
    job_directory_deleted="Job Directory Deleted",
    download_failed="Download Failed",
    base_model="Base Model",
    model_target="Target",
    model_load_finished="Model Load Finished",
    model_load_failed="Model Load Failed",
    training_job_submitted="Training Job Submitted",
    remote_training_finished="Remote Training Finished",
    remote_training_failed="Remote Training Failed",
    training_submission_cancelled="Training submission canceled by user. You can re-enter 'Run training' to fix settings.",
    modal_account_title="Modal Account",
    modal_secrets_title="Modal Secrets",
    modal_secret_configure="Configure Modal Secret:",
    modal_secret_back="Return to the main menu.",
    modal_secret_tokens_note="Tokens are sent to Modal only and are not written to config.toml or logs.",
    modal_secret_name="Modal Secret name:",
    modal_secret_password="HF_TOKEN for Modal Secret {secret_name}:",
    modal_secret_saved="Modal Secret {secret_name} is configured.",
    modal_secret_unchanged="No secret name entered; secret unchanged.",
    modal_secret_disabled="Disabled via {env}.",
    modal_secret_missing="MISSING",
    modal_secret_skipped="SKIPPED",
    modal_secret_unknown="UNKNOWN",
    modal_secret_ok="OK",
    modal_secret_disabled_via="disabled via {env}",
    modal_setup_missing_token="Token missing. Run `modal setup` to sign in.",
    modal_setup_complete="Modal setup finished. Refreshed status is shown below.",
    modal_setup_exit_code="Modal setup exited with code {code}. You can retry from the terminal or continue in the TUI.",
    modal_setup_browser_note="Starting Modal setup. Complete the browser flow, then return here.",
    modal_setup_browser_done="Modal setup finished. Refreshed status is shown below.",
    modal_setup_browser_retry="Modal setup exited with code {code}. You can retry from the terminal or continue in the TUI.",
    app_running="Modal app is running.",
    app_dashboard="Dashboard",
    function_call_id="Function Call ID",
    function_call_dashboard="Function Call",
    live_logs="Live logs",
    creating_volume_model="Creating the Modal Volume and downloading the model. This can take a while.",
    track_progress="Track progress with: modal app list && modal app logs <app-id>",
    detached_submit="Submitting remote training in detached mode.",
    detached_continue="The Modal app will keep running after the local process exits.",
    remote_allocating="Modal is allocating remote GPUs and checking container images.",
    synchronous_warning="This is synchronous mode; disconnecting the local process may cancel this run.",
    volume_name="Volume name",
    job_directory_to_delete="Job directory to delete",
    current_volume_name="Current volume name",
    new_volume_name="New volume name",
    volume_name_to_delete="Volume name to delete",
    config_for_output_lookup="Config for output lookup",
    output_name_override_hint="Used to read Model.output_name and locate /jobs/<job>/output/<output_name>.",
    local_destination_hint="Leave blank to use outputs/<job>/<output>.",
    no_secret_name="No secret name entered; secret unchanged.",
    no_token="No token entered; secret unchanged.",
    volume_list_title="Modal Volumes",
    session_billing_unavailable="Unavailable",
    session_billing_note="Billing report failed or timed out; session exit was not blocked.",
    session_billing_wait="Modal reports full billing intervals only; the latest partial hour may appear later.",
    session_total_cost="Total Reported Cost",
    session_rows="Rows",
    modal_profile="Modal Profile",
    billing_window="Billing Window",
    dashboard="Dashboard",
    session_start="Session Start",
    session_end="Session End",
    training_live_logs="Starting remote training. Live logs may stream below.",
    sync_base_model="Sync Base Model",
)


ZH = LanguagePack(
    code=LANG_ZH,
    app_title="NewBie-image LoRA 训练向导",
    app_subtitle="NewBie LoRA/LoKr 任务的引导式 Modal 工作流",
    main_menu="你想做什么？",
    main_run_training="启动训练",
    main_run_training_desc="上传或复用输入内容，并启动一个 Modal 训练任务。",
    main_download_output="下载任务输出",
    main_download_output_desc="从 Modal Volume 下载已完成的 adapter 输出目录。",
    main_load_model="加载模型到 Volume",
    main_load_model_desc="将基础模型 snapshot 下载到 /workspace/Models。",
    main_create_config="创建配置",
    main_create_config_desc="为新任务生成 LoRA 或 LoKr TOML 配置。",
    main_manage_volumes="管理 Volume",
    main_manage_volumes_desc="列出、重命名、删除或打开 Modal Volume。",
    main_quit="退出",
    main_quit_desc="退出，不再执行其他操作。",
    language_select_title="语言",
    language_select_prompt="请选择界面语言",
    language_select_desc="该偏好会本地保存，并在下次启动时复用。",
    language_save_confirm="记住这次选择的语言吗？",
    modal_setup_prompt="Modal token 缺失。现在要运行 `modal setup` 吗？",
    modal_setup_start="开始 Modal setup。完成浏览器流程后返回这里。",
    modal_setup_done="Modal setup 完成。下面会刷新状态。",
    modal_setup_retry="Modal setup 退出，返回码 {code}。你可以从终端重试，或者继续在 TUI 里操作。",
    modal_setup_skipped="已跳过 Modal setup。",
    returned_to_menu="已返回主菜单。",
    step_config_identity="步骤 1：配置身份",
    step_adapter="步骤 2：Adapter",
    step_training_defaults="步骤 3：训练默认值",
    step_launch_options="步骤 4：启动选项",
    step_summary="步骤 5：摘要",
    config_job_name="配置/任务名称",
    adapter_type="Adapter 类型",
    adapter_lokr_desc="Newbie-image 训练质量优先的推荐默认选项。",
    adapter_lora_desc="更小的 adapter，适合更快、更省显存的实验。",
    output_model_name="输出文件夹名称",
    training_resolution="训练分辨率",
    resolution_1024_desc="平衡的默认选项。",
    resolution_768_desc="显存占用更低，迭代更快。",
    resolution_1536_desc="细节更高，但 GPU 成本更高。",
    epochs="训练轮数",
    batch_size="批大小",
    learning_rate="学习率",
    training_config="训练配置",
    create_new_config="创建新配置",
    create_new_config_desc="通过引导式问题创建一个新的 LoRA/LoKr 任务配置。",
    modal_job_name="Modal 任务名称",
    first_upload_question="这个数据集是第一次为此任务上传吗？",
    first_upload_instruction="只有当该任务的配置和数据集已经在 Modal Volume 中时才选择 No。",
    dataset_directory="本地训练集目录",
    dataset_directory_instruction="选择包含图片和 caption 的文件夹，并上传给这个任务。",
    gpu="GPU",
    timeout_minutes="超时时间（分钟）",
    install_question="训练前设置或更新 trainer 依赖吗？",
    install_instruction="首次运行或 trainer 更新后建议选择 Yes。只有远程环境已准备好时才选择 No。",
    detach_question="本地终端断开后是否继续训练？",
    detach_instruction="Detached 模式会提交任务并让 Modal 继续运行，本地会返回控制权。",
    volume_management="Volume 管理",
    volume_list="列出 Volume",
    volume_list_desc="显示当前账号可用的 Modal Volume。",
    volume_delete="删除 Volume",
    volume_delete_desc="永久删除一个 Volume 及其中全部数据。",
    volume_rename="重命名 Volume",
    volume_rename_desc="不下载内容，直接修改 Volume 名称。",
    volume_dashboard="打开 Dashboard",
    volume_dashboard_desc="在 Modal dashboard 中打开所选 Volume。",
    back="返回",
    hf_repo="Hugging Face 仓库或 URL",
    revision="Revision",
    hf_secret="包含 HF_TOKEN 的 Modal Secret 名称",
    download_job_name="训练时使用的 Modal 任务名称",
    remote_output_override="远程输出目录覆盖值",
    local_destination="本地保存目录",
    session_closed="会话已结束",
    log_tail="日志尾部",
    preflight_check="起飞前检查",
    modal_account="Modal 账号",
    modal_secrets="Modal 密钥",
    upload_new="上传新数据",
    reuse_volume="复用 Volume",
    success="成功",
    failed="失败",
    available_folders="可用文件夹",
    no_volumes_found="没有找到 Volume。",
    no_jobs_found="/jobs 中没有找到任务目录。",
    create_new_job="创建新任务配置",
    create_new_job_desc="通过引导式问题创建一个新的 LoRA/LoKr 任务配置。",
    config_examples="示例或手写配置。",
    replace_existing_config="要替换已存在的配置 '{path}' 吗？",
    config_exists_retry="选择 No 可以输入不同的配置名。",
    allowed_name_chars="允许：字母、数字、点、下划线、连字符。",
    name_preview="名称预览：",
    use_slug_confirm="是否将 '{slug}' 作为 {noun} 的 slug？",
    use_slug_detail="你的输入会被保存为 '{slug}'，用于 Modal 路径、配置文件和本地文件夹。",
    dataset_folder_hint="包含图片和 caption 的文件夹，例如 ./data/my_images 或 D:\\datasets\\style",
    balanced_default="平衡的默认选项。",
    lower_memory="显存更低，迭代更快。",
    higher_detail="细节更高，但 GPU 成本更高。",
    official_lokr_default="Newbie-image 训练质量优先的推荐默认选项。",
    smaller_lora="更小的 adapter，适合更快、更省显存的实验。",
    images_per_step="每个 GPU step 的图片数。官方示例使用 4；高分辨率或显存受限时保持 1。",
    learning_rate_hint="优化器步长。官方示例：{official}；上游注释建议测试 1e-4 或 2e-4。",
    volume_delete_warning="这在 TUI 里无法撤销。",
    volume_rename_title="Volume 已重命名",
    volume_dashboard_opened="Dashboard 已打开",
    job_directory_deleted="任务目录已删除",
    download_failed="下载失败",
    base_model="基础模型",
    model_target="目标",
    model_load_finished="模型加载完成",
    model_load_failed="模型加载失败",
    training_job_submitted="训练任务已提交",
    remote_training_finished="远程训练完成",
    remote_training_failed="远程训练失败",
    training_submission_cancelled="用户已取消训练提交。你可以重新进入“启动训练”修正设置。",
    modal_account_title="Modal 账号",
    modal_secrets_title="Modal 密钥",
    modal_secret_configure="配置 Modal Secret：",
    modal_secret_back="返回主菜单。",
    modal_secret_tokens_note="Token 只会发送到 Modal，不会写入 config.toml 或日志。",
    modal_secret_name="Modal Secret 名称：",
    modal_secret_password="Modal Secret {secret_name} 的 HF_TOKEN：",
    modal_secret_saved="Modal Secret {secret_name} 已配置。",
    modal_secret_unchanged="未输入 Secret 名称；Secret 保持不变。",
    modal_secret_disabled="已通过 {env} 禁用。",
    modal_secret_missing="缺失",
    modal_secret_skipped="跳过",
    modal_secret_unknown="未知",
    modal_secret_ok="正常",
    modal_secret_disabled_via="通过 {env} 禁用",
    modal_setup_missing_token="Token 缺失。运行 `modal setup` 完成登录。",
    modal_setup_complete="Modal setup 完成。下面会刷新状态。",
    modal_setup_exit_code="Modal setup 退出，返回码 {code}。你可以从终端重试，或者继续在 TUI 里操作。",
    modal_setup_browser_note="开始 Modal setup。完成浏览器流程后返回这里。",
    modal_setup_browser_done="Modal setup 完成。下面会刷新状态。",
    modal_setup_browser_retry="Modal setup 退出，返回码 {code}。你可以从终端重试，或者继续在 TUI 里操作。",
    app_running="Modal app 正在运行。",
    app_dashboard="Dashboard",
    function_call_id="Function Call ID",
    function_call_dashboard="Function Call",
    live_logs="实时日志",
    creating_volume_model="正在创建 Modal Volume 并下载模型，这可能需要一些时间。",
    track_progress="查看进度：modal app list && modal app logs <app-id>",
    detached_submit="正在以 detached 模式提交远程训练。",
    detached_continue="本地进程退出后，Modal app 会继续运行。",
    remote_allocating="Modal 正在分配远程 GPU 并检查容器镜像。",
    synchronous_warning="这是同步模式；断开本地进程可能会取消本次运行。",
    volume_name="Volume 名称",
    job_directory_to_delete="要删除的任务目录",
    current_volume_name="当前 Volume 名称",
    new_volume_name="新的 Volume 名称",
    volume_name_to_delete="要删除的 Volume 名称",
    config_for_output_lookup="用于输出查找的配置",
    output_name_override_hint="用于读取 Model.output_name，并定位 /jobs/<job>/output/<output_name>。",
    local_destination_hint="留空则使用 outputs/<job>/<output>。",
    no_secret_name="未输入 Secret 名称；Secret 保持不变。",
    no_token="未输入 Token；Secret 保持不变。",
    volume_list_title="Modal Volumes",
    session_billing_unavailable="不可用",
    session_billing_note="Billing 报告失败或超时；会话退出未被阻塞。",
    session_billing_wait="Modal 只会报告完整计费区间；最新的未满一小时部分可能稍后出现。",
    session_total_cost="已报告总费用",
    session_rows="行数",
    modal_profile="Modal Profile",
    billing_window="计费窗口",
    dashboard="Dashboard",
    session_start="会话开始",
    session_end="会话结束",
    training_live_logs="开始远程训练，下面可能会输出实时日志。",
    sync_base_model="加载模型到 Volume",
)


PACKS = {LANG_EN: EN, LANG_ZH: ZH}


def normalize_lang(value: str | None) -> str:
    lang = (value or "").strip().lower()
    return lang if lang in SUPPORTED_LANGS else DEFAULT_LANG


def get_pack(lang: str | None) -> LanguagePack:
    return PACKS[normalize_lang(lang)]
