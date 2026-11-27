# InternVLA-A1.5 + VLASH 动态传送带翻包数据预处理、训练与推理指南

本教程说明如何在 `/home/jiangxirui/g1/InternVLA-A-series` 中将 Galbot 真机遥操作 `.mcap` 数据转换为 LeRobot v3.0 格式、计算 VLASH 相对动作统计量、在 Jetson Orin / GPU 上实测推理延迟，并启动 `InternVLA-A1.5 + VLASH + WAN2.2-5B` 联合微调与异步推理部署。

## 1. 环境与离线权重

- **Conda 环境**：`/mnt/data3/jiangxirui/conda_envs/internvla_a15`（Python 3.11 + PyTorch 2.10 + `transformers==5.2.0` + `transformers_replace` 补丁）。
- 若重新安装过 `transformers` 或更新过源码，请重新执行补丁脚本：
  ```bash
  bash /home/jiangxirui/g1/InternVLA-A-series/scripts/setup_env_and_patch.sh
  ```
- **本地离线权重目录**（均位于 `/mnt/data3/`）：
  - `/mnt/data3/InternVLA-A1.5-base`
  - `/mnt/data3/Wan2.2-TI2V-5B`
  - `/mnt/data3/Qwen3.5-2B`
  - `/mnt/data3/fast_tokenizer`

如需补全或校验权重完整性：
```bash
/mnt/data3/jiangxirui/conda_envs/internvla_a15/bin/python \
  /home/jiangxirui/g1/InternVLA-A-series/tools/parcel_flip/download_weights.py --skip-existing
```

## 2. 一体化 MCAP 数据预处理与 VLASH 相对动作统计

使用单一开箱即用脚本 [`tools/parcel_flip/convert_mcap_to_internvla_lerobot_v3.py`](../tools/parcel_flip/convert_mcap_to_internvla_lerobot_v3.py)（零 `torch`/`lerobot`/`ffprobe` 硬依赖），一步完成转换、相对统计量计算与端到端校验（`--step all`）：

- **左臂优先 16 维布局**：**`left_arm[7] + left_gripper[1] + right_arm[7] + right_gripper[1]`**，与 InternVLA-A1.5 预训练先验及 `galbot_one_golf.yaml` 严格一致；严格校验传感器组存在性与关节维度；
- **Unified 单指令默认**：默认 `--task-mode unified` 为全部子目录统一绑定 `"Flip the parcel so the label faces up"`（无外部朝向分类器时由 VLA 看图自主决策翻包或直推；如需按目录区分可切 `--task-mode per_folder`）；
- **30Hz 均匀重采样 + 因果零阶保持（Causal ZOH）**：在三路相机与传感器的重叠时间窗口上按均匀 `30Hz` 重采样，target 流使用 Causal ZOH + 启动宽限期 + 尾部保持救援（`hold_eps_rad=0.05`）+ 中段空洞切段；
- **`224x224` Letterbox + 高码率编码校验**：采用 `INTER_LINEAR`（匹配推理端 `resize_with_pad`）+ `libx264 crf=18 g=2 preset=veryfast`，原子写盘并逐视频校验帧数、分辨率、`PTS < 1e-4s`、首帧 MAE 与黑边均值；
- **一步输出 VLASH 相对统计量（`chunk_size=30`）**：自动写出精确 `float64` 全局统计量（`action` 为相对动作且默认屏蔽尾部填充步以对齐 `mask_action_pad: true`，`observation.state` 与夹爪 `[7, 15]` 保持绝对值，并保留 `meta/stats.json.abs_bak`）。

Windows 本机直接运行示例：
```powershell
python tools\parcel_flip\convert_mcap_to_internvla_lerobot_v3.py `
  --source-root D:\dynamic_parcel_flip_data `
  --output-dir D:\dynamic_rigid_parcel_flip_internvla_lerobot_v3_rel30 `
  --workers 4
```

Linux 服务器运行示例：
```bash
/mnt/data3/jiangxirui/conda_envs/internvla_a15/bin/python \
  /home/jiangxirui/g1/InternVLA-A-series/tools/parcel_flip/convert_mcap_to_internvla_lerobot_v3.py \
  --source-root /mnt/data3/jiangxirui/dynamic_parcel_flip_data \
  --output-dir /mnt/data3/jiangxirui/dynamic_rigid_parcel_flip_internvla_lerobot_v3_rel30 \
  --workers 4
```

> 如需在已有转换结果上仅重算 `stats.json` 并重新校验，可运行 `--step finalize,verify --force`（无需重编码视频）；若曾用旧版脚本转换过，请加 `--force` 全量重建。

## 3. 内置自检（无需真机 MCAP）

```bash
python tools/parcel_flip/convert_mcap_to_internvla_lerobot_v3.py --self-test
```

## 4. 建议在开训前先做 Orin / GPU 3 相机延迟基准测试

用基座权重直接实测 `InternVLAA15Optimized`（SDPA + CUDA Graph，3 视角，10 步流匹配）的端到端推理耗时，再据此决定是否收窄 `vlash_min_delay_steps` / `vlash_max_delay_steps`：

```bash
/mnt/data3/jiangxirui/conda_envs/internvla_a15/bin/python \
  /home/jiangxirui/g1/InternVLA-A-series/tools/parcel_flip/benchmark_orin_latency.py \
  --ckpt-path /mnt/data3/InternVLA-A1.5-base \
  --vlm-path /mnt/data3/Qwen3.5-2B \
  --num-views 3 \
  --chunk-size 30 \
  --fps 30
```

## 5. 启动 InternVLA-A1.5 + VLASH 训练

默认配置 [`configs/parcel_flip/internvla_a15_vlash_conveyor_delta_3cam.yaml`](../configs/parcel_flip/internvla_a15_vlash_conveyor_delta_3cam.yaml) 包含：
- **标准 3 路相机输入**：`enable_head_history=false`（`head_right`, `left_arm`, `right_arm`），与预训练一致且最小化 Orin 端推理延迟；
- **`mean_std` 归一化对齐预训练**：保持 `InternVLA-A1.5-base` 的流匹配信噪比与状态离散化刻度；
- **VLASH 显式延迟条件**：`vlash_condition_on_delay=true`，在 VLM 提示词中注入 `; Delay: <δ>`，解决机械臂静止待命拦截移动包裹时的延迟不可辨识问题；`vlash_zero_delay_prob=0.1`；仅在 $\delta=0$ 时启用 `state_dropout_prob=0.1`；
- **固定跨度 WAN2.2-5B 潜空间前瞻**：`wan_video_span_steps=42`（固定 42 帧跨度，不再随随机 $\delta$ 变速拉伸），并对 WAN2.2 DiT blocks 开启 gradient checkpointing 防止单卡 OOM；
- **动作 Loss 精确掩码**：`mask_action_pad=true` 屏蔽 episode 尾部重复 clamp 帧，`padded_action_dim_weight=0.1` 防止 16 维补零通道稀释真实动作梯度同时稳住流匹配 ODE；
- **显式 Letterbox 黑边保护**：`letterbox_pad_px=28`（对应 640×480 → 224×168 上下各 28px 黑边），避免 ColorJitter 污染黑边；
- **环境适配**：`video_backend: pyav`，`use_sdpa: false` 保证数值稳定，`WAN2.2` 无 `flash_attn` 时自动走已修复 dtype 的 PyTorch SDPA 回退。

先导出 SwanLab API Key（如需云端同步），再启动单卡训练（默认 GPU 4）：
```bash
export SWANLAB_API_KEY="<your-swanlab-api-key>"
bash /home/jiangxirui/g1/InternVLA-A-series/launch/parcel_flip/train_internvla_a15_vlash_conveyor.sh
```

多卡 DDP 启动（例如 4 卡，建议同步把学习率提到 `5e-5` 对齐官方 BS=32 配方）：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,4 NUM_PROCESSES=4 \
  bash /home/jiangxirui/g1/InternVLA-A-series/launch/parcel_flip/train_internvla_a15_vlash_conveyor.sh \
  --policy.optimizer_lr=5e-5 --policy.scheduler_decay_lr=5e-6
```

## 6. VLASH 异步推理部署（`InternVLAA15Optimized` + Future-State Roll-Forward）

使用 [`tools/parcel_flip/vlash_async_policy.py`](../tools/parcel_flip/vlash_async_policy.py) 加载微调后的 checkpoint（自动丢弃 `wan_video_model`、启用 CUDA Graph、按 `train_config.json` 自动匹配 `mean_std`/`q01_q99`、`Delay: <δ>`、`Output: <Action>`，并支持伺服滞后补偿与可选的 Galbot SDK 右臂在前维度重排）：

```python
from tools.parcel_flip.vlash_async_policy import VLASHAsyncInternVLAPolicy

runner = VLASHAsyncInternVLAPolicy.from_checkpoint(
    ckpt_dir="/mnt/data3/jiangxirui/train_outputs/.../pretrained_model",
    vlm_model_path="/mnt/data3/Qwen3.5-2B",
    fps=30.0,
    servo_lag_steps=2,          # 实测 30Hz 下 target 超前 sensor 约 2~3 帧（67~100ms）
    action_quant_ratio=1,       # >1 时启用 VLASH Action Quantization 提速
    sdk_right_arm_first=True,   # 若 Galbot SDK 要求 right_arm 在前则自动做输入输出重排
)
```
