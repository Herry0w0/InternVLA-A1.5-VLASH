# InternVLA-A1.5 + VLASH Jetson AGX Orin 推理性能优化报告

本报告记录 **InternVLA-A1.5 + VLASH**（3B 参数：ViT 331.4M + Qwen3.5-2B VLM 1373.3M + Action Expert 460.4M）在 **Jetson AGX Orin 64GB（SM 8.7）** 上的推理瓶颈分析、分步优化思路、仓库架构改动以及实测延迟与精度验证结果。

最终在 **3 路相机 `224×224`、`chunk_size=30`、10 步流匹配去噪** 的完整闭环条件下，将单次 Orin 推理延迟从原版 **16,297.8 ms**（基线修复后 **1,792.8 ms**）压降至 **265–274 ms**（5 步去噪 **212.3 ms**），余弦相似度保持 **`0.999885`**，成功突破 `< 300 ms` 目标并优于 Pi0.5 的 340 ms 表现。

---

## 1. 硬件环境与测试基准

- **目标平台**：Jetson AGX Orin 64GB（L4T `5.10.216-tegra` aarch64，12 核 ARM CPU，16 SMs，`MAXN` 模式，GPU 锁频 `1300.5 MHz`）。
- **运行栈**：Python 3.11 + CUDA 11.8 + Orin 编译版 PyTorch `2.4.0a0`（BF16 支持，无 cuDNN、无 FlashAttention、无 CPU LAPACK）+ `transformers==5.2.0` + `FlashRT` SM87 算子库 + 自研融合扩展 `internvla_orin_ops`。
- **统一测试配置**：[`tools/parcel_flip/benchmark_orin_latency.py`](../tools/parcel_flip/benchmark_orin_latency.py)，`--num-views 3 --chunk-size 30 --num-inference-steps 10 --fps 30.0`，`action_loss_only=True`（推理阶段剔除 WAN2.2 视频分支）。

---

## 2. 核心瓶颈诊断

对原版 `InternVLAA15Optimized` 在 Orin 上做逐模块剖析，定位出六大核心瓶颈：

1. **无 cuDNN 下的 `Conv3d` 灾难性回退（13.2 s）**：`Qwen3_5VisionPatchEmbed` 使用无偏置、步长等于核大小的 `nn.Conv3d`，在无 cuDNN 的 Orin PyTorch 上退化为极慢实现，单这一层耗时 `13,218 ms`。
2. **Gated DeltaNet (GDN) 的 63 步串行 Python 循环**：`torch_chunk_gated_delta_rule` 内部存在 63 次顺序矩阵迭代，且无 `flash-linear-attention` / `causal-conv1d` 扩展时回退到大量碎小 PyTorch 算子，launch 与 DRAM 访存开销巨大。
3. **Action Expert 每步重复计算 50 个 Learnable Tokens**：流匹配 10 步去噪中，前 50 个可学习 token 不依赖加噪动作或时间步，却在每步都重复前向（每步 `80 tokens = 50 LT + 30 Action`，10 步共多算 500 tokens）。
4. **Orin 内存带宽受限与 GEMM 维度未对齐**：
   - Action Expert（`M=30`）属于典型 skinny GEMM，BF16 权重读取受限于 Orin 统一内存带宽；
   - 默认 VLM 前缀长度 `L=416` 不是 128 的整数倍，触发 cuBLAS 慢速 tile 回退，仅 VLM MLP 就多消耗 `42.6 ms`（`95.9 ms` vs `L=384` 的 `53.3 ms`）。
5. **逐元素胶水算子过多**：`SiLU*Mul`、`QKVZ` 切分、GDN WY 预处理、`RMSNorm + RoPE + KV 写入`、ViT `Residual + LayerNorm` / `GELU` 等分散为数百个小 CUDA kernel，造成大量 HBM 读写往返。
6. **高 CPU 调度抖动与环境陷阱**：ARM CPU 发射 Python/PyTorch 算子延迟高；此外原版 `torch.zeros(...).to(value)` 破坏 CUDA Graph 捕获，`resize_token_embeddings` 默认协方差初始化在 ARM CPU 上耗时 20 分钟且触发无 LAPACK 崩溃。

---

## 3. 优化思路与分步演进

围绕 **"消除框架回退 → 减少冗余计算 → 替换核心重算子为 SM87 定制内核 → 权重量化降带宽 → 融合胶水算子与维度对齐 → 全链路静态图化"** 六个阶段推进：

| 阶段 | 核心优化动作 | 10 步 Mean / P50 | 10 步 P99 | 5 步 Mean | vs `standard` 余弦相似度 |
|---|---|---:|---:|---:|---:|
| 0. 原版开箱 | 原始 `optimized` 后端（无 cuDNN `Conv3d`） | `16,297.8 ms` | — | — | `1.0000` |
| 1. 基线修复 | `PatchEmbed` `Conv3d → F.linear`（`13,218 ms → 0.19 ms`） | `1,792.8 ms` / `1,792.7 ms` | `1,807.0 ms` | `1,283.7 ms` | `1.0000` |
| 2. 纯 PyTorch 数学加速 | 开启 TF32 + GDN 6 步幂零倍增向量化 + Action Expert BF16 | `833.6 ms` | — | `643.2 ms` | `0.9999+` |
| 3. FlashRT v1 引擎 | FlashRT WY GDN + FA2 + `causal_conv1d` + 50-LT 状态缓存 + W8A16 + 双 CUDA Graph（`L=416`） | `374.5 ms` | — | — | `0.999881` |
| 4. FlashRT v2 极限优化 | 自研 `internvla_orin_ops` 8 个 SM87 融合算子 + VLM 128 对齐分桶（`L=384`） | **`274.1 ms` / `274.5 ms`** (`3.64 Hz`) | **`282.6 ms`** | **`212.3 ms`** (`P99 218.2 ms`) | **`0.999885`** (`5步: 0.999767`) |
| 5. 便携包洁净实测 | 脱离宿主机环境（`env -i`）运行自包含部署包 | **`265.0 ms` / `265.2 ms`** (`3.77 Hz`) | **`267.7 ms`** | — | 同引擎零差异 |

---

## 4. 关键技术实现与仓库改动清单

### 4.1 通用训练/推理架构修复

- **[`src/lerobot/policies/internvla_a1_5/transformers_replace/models/qwen3_5/modeling_qwen3_5.py`](../src/lerobot/policies/internvla_a1_5/transformers_replace/models/qwen3_5/modeling_qwen3_5.py)**：
  - `Qwen3_5VisionPatchEmbed.forward` 用数学等价的 `F.linear` 替代 `nn.Conv3d`；
  - `torch_chunk_gated_delta_rule` 用 6 步精确幂零倍增 $(I+A)(I+A^2)\cdots(I+A^{32})$ 替代 63 步串行循环（验证 `max_abs_diff = 5.82e-11`）；
  - 修复 3 处 `torch.zeros(...).to(value)` 为 `torch.zeros(..., device=..., dtype=...)`，解除 CUDA Graph 捕获阻塞。
- **[`src/lerobot/policies/internvla_a1_5/action_tokens.py`](../src/lerobot/policies/internvla_a1_5/action_tokens.py)**：
  - `model.resize_token_embeddings(target_size, mean_resizing=False)`，跳过冗余 CPU 协方差计算（新行紧接着由 `_init_new_rows` 覆写），消除启动卡死与 LAPACK 报错。
- **[`src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py`](../src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5.py)** 与 **[`configuration_internvla_a1_5.py`](../src/lerobot/policies/internvla_a1_5/configuration_internvla_a1_5.py)**：
  - 修复非 autocast 下 `denoise_step` / `denoise_step_full` 中 `suffix_embs` dtype 不一致问题；
  - 正式注册 `inference_backend="flashrt"` 及 `flashrt_w8a16_action_expert`、`flashrt_use_cuda_graph` 参数。

### 4.2 FlashRT SM87 极速引擎与自研融合算子

- **[`src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5_flashrt.py`](../src/lerobot/policies/internvla_a1_5/modeling_internvla_a1_5_flashrt.py)**（`InternVLAA15FlashRT`）：
  1. **ViT 批处理 + 静态位置/RoPE 缓存**：3 路图像合并 batch，缓存 2D 位置编码与 RoPE，融合 `Residual+LayerNorm`、`QKV+RoPE+Transpose` 与 `GELU`；
  2. **VLM 128 对齐动态分桶**：根据 `valid_len` 自动向上取整到 128 的倍数（常规 3 视角输入 `valid_len <= 384 → L=384`），彻底规避 `L=416` 的 cuBLAS tile 劣化并减少 1 个 GDN WY chunk；
  3. **核心算子全面替换为 FlashRT SM87 内核**：
     - 线性注意力层（18 层）：合并 `QKVZBA` 投影 + FlashRT `causal_conv1d_qwen36_bf16` + 5 阶段 `cublaslt_packed` WY GDN（注意：SM87 上 FlashRT `mma_fla` 输出核在 `H=16` 时数值异常，务必走 `cublaslt_packed` 路径，实测 `out_cos = 0.99992–0.99997`）；
     - 全注意力层（6 层，`head_dim=256`）：合并 `QKV` 投影 + FlashRT FA2 `fwd_bf16_causal`（VLM）与 `fwd_bf16_seqused`（Action Expert，零 4D mask，按精确 `valid_len` 截断）；
     - MLP 层：合并 `Gate+Up` 投影，减少 GEMM 启动次数。
  4. **50-Learnable-Token 数学等价状态缓存**：将 50 个前缀不变的可学习 token 移至 VLM 之后执行一次，缓存其 KV Cache、Conv1d 状态与 GDN 隐状态；10 步去噪每步仅运行 30 个动作 token（序列长度 `80 → 30`，单 chunk 完成 GDN）；
  5. **Action Expert W8A16 仅权重量化**：采用 FlashRT `w8a16_skinny_gemm`（权重量化为 INT8 + per-channel scale，激活维持 BF16），减半权重访存带宽同时避免 W8A8 带来的 VLA 精度崩塌；
  6. **全链路双 CUDA Graph**：
     - **Graph 1**：`ViT + VLM 24L + 50-LT` 前缀编码；
     - **Graph 2**：完整 $N$ 步去噪循环（含预计算正弦时间步嵌入与状态回滚），将每步调度开销压至近零。

- **[`src/lerobot/policies/internvla_a1_5/csrc/internvla_orin_ops.cu`](../src/lerobot/policies/internvla_a1_5/csrc/internvla_orin_ops.cu)**（编译产物 `internvla_orin_ops.cpython-311-aarch64-linux-gnu.so`）：
  - 使用 `nvcc -O3 --use_fast_math -std=c++17 -arch=sm_87` 直接基于 CUDA + pybind11 原始设备指针编译，**零 PyTorch C++ ABI 耦合**；
  - 提供 8 个融合算子：`silu_mul_split_bf16`、`split_qkvz_bf16`、`gdn_prep_wy_bf16`、`qkv_split_norm_rope_kvwrite_bf16`、`attn_gate_sigmoid_from_qkv_bf16`、`residual_add_layernorm_bf16`、`vit_qkv_rope_transpose_d64_bf16`、`gelu_tanh_inplace_bf16`。

---

## 5. 精度校验、鲁棒性与 VLASH 配置结论

1. **数值一致性（vs `standard` PyTorch）**：
   - `steps=10`：`cos = 0.999885`，`mean_abs_diff = 0.01251`，`max_abs_diff = 0.04907`；
   - `steps=5`：`cos = 0.999767`，`mean_abs_diff = 0.01813`，`max_abs_diff = 0.09085`。
2. **动态提示词长度压测**：
   - `valid_len=373`（`cos=0.999880`，命中 `L=384` 图）→ `valid_len=377`（`cos=0.999876`，零重捕获复用 `L=384` 图）→ `valid_len=389`（`cos=0.999887`，自动分桶 `L=512` 图）。
3. **VLASH 延迟超参确认（[`configs/parcel_flip/internvla_a15_vlash_conveyor_delta_3cam.yaml`](../configs/parcel_flip/internvla_a15_vlash_conveyor_delta_3cam.yaml)）**：
   - 空闲 Orin 实测 `265–274 ms @ 10步`（30 Hz 下约 `8–9` 步）或 `212 ms @ 5步`（约 `6–7` 步）；
   - 参考同机 Pi0.5 实测经验，真机同时运行控制/相机 SDK 会引入 `+75–85 ms` DRAM 带宽争用，预计真机闭环单次推理约 `280–360 ms`（`9–11` 步）；
   - 因此训练配置保留 **`vlash_max_delay_steps: 12`**（对应 `400 ms @ 30 Hz`）与 `wan_video_span_steps: 42` 是当前最稳妥的区间，既覆盖真机抖动余量，又无需放大预测跨度。

---

## 6. 复现与同型号 Orin 开箱即用部署

### 6.1 便携部署压缩包

已同步构建最小化、可重定位的 Orin 运行时压缩包（`internvla_a15_vlash_orin_flashrt_bundle.tar.gz`，压缩后 `1.3 GB`，解压 `2.8 GB`），包含 Python 3.11、82 个必需 CUDA/系统运行库、Orin PyTorch 2.4、已打补丁的 `transformers==5.2.0`、精简 `flash_rt`（`flash_rt_kernels.so` + `flash_rt_fa2.so`）、预编译 `internvla_orin_ops.so` 及完整代码仓库。在 `env -i PATH=/usr/bin:/bin HOME=/tmp` 洁净环境下已完成零外部路径泄漏验证与 `265.0 ms` 端到端实测。

在新机器解压后执行：

```bash
tar -xzf internvla_a15_vlash_orin_flashrt_bundle.tar.gz
cd internvla_a15_vlash_orin_flashrt_bundle

# 1. 一键环境与 SM87 算子自检
./run.sh verify_orin_env.py

# 2. 端到端延迟与数值一致性实测（默认自动使用 flashrt 后端）
./run.sh InternVLA-A1.5-VLASH/tools/parcel_flip/benchmark_orin_latency.py \
    --ckpt-path /path/to/InternVLA-A1.5-ckpt \
    --vlm-path /path/to/Qwen3.5-2B \
    --backend flashrt --verify-parity \
    --num-views 3 --chunk-size 30 --num-inference-steps 10 --fps 30.0
```

### 6.2 在异步策略代码中启用 FlashRT

[`tools/parcel_flip/vlash_async_policy.py`](../tools/parcel_flip/vlash_async_policy.py) 已内置 `flashrt` 后端支持：

```python
from tools.parcel_flip.vlash_async_policy import VLASHAsyncInternVLAPolicy

runner = VLASHAsyncInternVLAPolicy.from_checkpoint(
    ckpt_dir="/path/to/finetuned_ckpt",
    vlm_model_path="/path/to/Qwen3.5-2B",
    device="cuda",
    inference_backend="flashrt",
    w8a16_action_expert=True,
    use_cuda_graph=True,
    fps=30.0,
    auto_warmup=True,
)
```

> 真机闭环建议：上机运行前开启 `sudo nvpmodel -m 0 && sudo jetson_clocks`，并在桌面/传送带定点操作场景挂起导航栈，以尽量降低 Orin DRAM 带宽争用。
