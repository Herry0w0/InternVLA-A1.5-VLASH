"""FlashRT-accelerated InternVLA-A1.5 inference backend for Jetson AGX Orin (SM87).

End-to-end Orin acceleration stack (targeting <270 ms idle @ 10 steps / <210 ms @ 5 steps):
  1. Vision Tower (Qwen3.5 ViT, 24 layers):
     - Mathematically exact F.linear patch_embed (eliminates no-cuDNN Conv3d fallback).
     - Precomputed static 2D position embeddings & rotary cos/sin per grid_thw.
     - Fused SM87 residual_add_layernorm_bf16, vit_qkv_rope_transpose_d64_bf16,
       gelu_tanh_inplace_bf16, and batched multi-view forward (zero cu_seqlens syncs).
  2. VLM Language Backbone (Qwen3.5-2B, 24 layers = 18 GDN + 6 FullAttn, BF16):
     - Automatic 128-aligned sequence bucketing (e.g. 380 valid tokens -> L=384 = 3x128)
       to hit peak SM87 cuBLAS Tensor-Core tile waves (-42.6 ms vs unaligned L=416).
     - Fused QKVZBA / QKV / Gate-Up linear projections with zero intermediate split copies.
     - FlashRT causal_conv1d_qwen36_bf16 + custom fused SM87 gdn_prep_wy_bf16
       + 5-stage cuBLASLt Tensor-Core WY GDN.
     - Custom fused qkv_split_norm_rope_kvwrite_bf16 + FlashRT FA2 fwd_bf16_causal
       + attn_gate_sigmoid_from_qkv_bf16 + silu_mul_split_bf16.
  3. Action Expert (460M, 24 layers = 18 GDN + 6 FullAttn):
     - Mathematically exact 50-learnable-token KV + Conv1d + GDN state caching:
       runs prefix-invariant learnable tokens ONCE per inference instead of 10x,
       shrinking per-step denoising length from 80 -> 30 tokens (-62.5% tokens, 1 GDN chunk).
     - SM87 W8A16 weight-only INT8 skinny GEMM (activations stay BF16, cos >= 0.99996).
     - FlashRT FA2 fwd_bf16_seqused over exact valid prefix + cached LT + action KV
       (zero 4D attention mask, zero padding KV reads).
  4. Full-pipeline CUDA Graph capture & position_ids caching:
     - Graph 1: ViT + prefix embedding + VLM LM + 50-LT Action-Expert prefill.
     - Graph 2: All N flow-matching denoising steps unrolled inside one CUDA Graph
       with precomputed sinusoidal timestep embeddings.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import InternVLAA15Config
from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import (
    InternVLAA15,
    create_sinusoidal_pos_embedding,
)

logger = logging.getLogger(__name__)


def _import_flash_rt():
    try:
        from flash_rt import flash_rt_fa2 as fa2
        from flash_rt import flash_rt_kernels as fk
    except Exception as exc:
        raise RuntimeError(
            "inference_backend='flashrt' requires FlashRT compiled CUDA extensions "
            "(flash_rt.flash_rt_kernels and flash_rt.flash_rt_fa2)."
        ) from exc
    try:
        from lerobot.policies.internvla_a1_5 import internvla_orin_ops as orin_ops
    except Exception as exc:
        raise RuntimeError(
            "inference_backend='flashrt' requires compiled internvla_orin_ops SM87 CUDA extension."
        ) from exc
    return fk, fa2, orin_ops


def _quantize_w8a16_rowwise(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Symmetric per-output-channel INT8 weight-only quantization for w8a16_skinny_gemm."""
    w_f32 = weight.detach().float()
    scale = w_f32.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    w_i8 = (w_f32 / scale[:, None]).round().clamp(-127.0, 127.0).to(torch.int8).contiguous()
    return w_i8, scale.to(torch.float32).contiguous()


def _select_w8a16_cfg(M: int, N: int, K: int) -> tuple[int, int] | None:
    """Select tuned (variant, split_k) for SM87 w8a16_skinny_gemm, or None to fallback to BF16."""
    if M <= 0 or M > 64:
        return None
    if N % 64 != 0 or K % 64 != 0:
        return None
    # Tuned on Orin SM87 (16 SMs) at M=30 for Action Expert shapes
    if N >= 5120 and N % 128 == 0 and K <= 1024:
        return (4, 1) if N != 6144 else (0, 1)
    if N == 1024 and K == 2048:
        return (4, 4) if (N % 128 == 0 and K % 256 == 0) else (0, 4)
    if N == 1024 and K == 3072:
        return (0, 4)
    variant = 4 if (N % 128 == 0 and N >= 2048) else 0
    bn = 128 if variant == 4 else 64
    blocks_n = N // bn
    split_k = 1
    if blocks_n < 32:
        for cand in (4, 2):
            if K % (64 * cand) == 0 and blocks_n * cand <= 128:
                split_k = cand
                break
    return (variant, split_k)


class _PackedLinear:
    """BF16 or SM87 W8A16 weight-only linear operator writing into a preallocated output buffer."""

    __slots__ = ("N", "K", "use_w8a16", "w_bf16", "w_i8", "scale", "variant", "split_k")

    def __init__(self, weight: Tensor, *, use_w8a16: bool = False, max_m: int = 30) -> None:
        self.N, self.K = int(weight.shape[0]), int(weight.shape[1])
        cfg = _select_w8a16_cfg(max_m, self.N, self.K) if use_w8a16 else None
        if cfg is not None:
            self.use_w8a16 = True
            self.w_i8, self.scale = _quantize_w8a16_rowwise(weight)
            self.w_bf16 = None
            self.variant, self.split_k = cfg
        else:
            self.use_w8a16 = False
            self.w_bf16 = weight.detach().to(torch.bfloat16).contiguous()
            self.w_i8 = None
            self.scale = None
            self.variant, self.split_k = -1, 1

    def forward(self, fk, a_bf16: Tensor, out_bf16: Tensor, partials: Tensor | None, stream: int) -> Tensor:
        if self.use_w8a16:
            M = a_bf16.shape[0]
            rc = fk.w8a16_skinny_gemm(
                a_bf16.data_ptr(),
                self.w_i8.data_ptr(),
                self.scale.data_ptr(),
                out_bf16.data_ptr(),
                partials.data_ptr() if partials is not None else 0,
                M,
                self.N,
                self.K,
                self.variant,
                self.split_k,
                stream,
            )
            if rc != 0:
                raise RuntimeError(
                    f"w8a16_skinny_gemm failed (rc={rc}) for M={M}, N={self.N}, K={self.K}, "
                    f"variant={self.variant}, split_k={self.split_k}"
                )
            return out_bf16
        torch.matmul(a_bf16, self.w_bf16.t(), out=out_bf16)
        return out_bf16


class InternVLAA15FlashRT(InternVLAA15):
    """Extreme Orin SM87 FlashRT backend for InternVLA-A1.5."""

    def __init__(self, config: InternVLAA15Config):
        if not config.action_loss_only:
            raise ValueError(
                "InternVLAA15FlashRT requires action_loss_only=True so the WAN branch is not loaded."
            )
        super().__init__(config)
        self._fk, self._fa2, self._ops = _import_flash_rt()
        self._packed: bool = False
        self._vit_static_cache: dict[tuple[tuple[int, int, int], ...], dict[str, Any]] = {}
        self._pos_ids_cache: dict[tuple, tuple[Tensor, Tensor]] = {}
        self._pipelines: dict[tuple[int, int, int, int, int], dict[str, Any]] = {}

    def train(self, mode: bool = True):
        if mode:
            raise RuntimeError("InternVLAA15FlashRT is an inference-only backend.")
        return super().train(False)

    def invalidate_packed_weights(self) -> None:
        self._packed = False
        self._vit_static_cache.clear()
        self._pos_ids_cache.clear()
        self._pipelines.clear()

    def prepare_flashrt(self, device: torch.device | None = None) -> None:
        """Pack & fuse weights for ViT, VLM LM, and Action Expert on the target CUDA device."""
        if device is None:
            device = next(self.parameters()).device
        if device.type != "cuda":
            raise RuntimeError("InternVLAA15FlashRT requires a CUDA device.")

        torch.backends.cuda.matmul.allow_tf32 = True
        vlm = self.qwen3_5_with_expert.qwen3_5
        lm = vlm.language_model
        act = self.qwen3_5_with_expert.action_expert
        vis = vlm.visual
        vlm_cfg = vlm.config.text_config
        act_cfg = act.config

        self._vlm_hidden = int(vlm_cfg.hidden_size)
        self._vlm_inter = int(vlm_cfg.intermediate_size)
        self._act_hidden = int(act_cfg.hidden_size)
        self._act_inter = int(act_cfg.intermediate_size)
        self._num_layers = int(vlm_cfg.num_hidden_layers)
        self._layer_types = [layer.layer_type for layer in lm.layers]
        self._rms_eps = float(vlm_cfg.rms_norm_eps)
        self._head_dim = int(vlm_cfg.head_dim)
        self._num_q_heads = int(vlm_cfg.num_attention_heads)
        self._num_kv_heads = int(vlm_cfg.num_key_value_heads)
        self._rope_dim = int(self._head_dim * float(vlm_cfg.rope_parameters.get("partial_rotary_factor", 1.0)))
        self._lin_num_k_heads = int(vlm_cfg.linear_num_key_heads)
        self._lin_num_v_heads = int(vlm_cfg.linear_num_value_heads)
        self._lin_head_dim = int(vlm_cfg.linear_key_head_dim)
        self._lin_conv_dim = self._lin_num_k_heads * self._lin_head_dim * 2 + self._lin_num_v_heads * self._lin_head_dim
        self._lin_qk_group = self._lin_num_v_heads // self._lin_num_k_heads
        self._image_token_id = int(vlm.config.image_token_id)

        # ---- 1. Pack ViT weights ----
        self._vit_depth = len(vis.blocks)
        self._vit_hidden = int(vis.config.hidden_size)
        self._vit_inter = int(vis.config.intermediate_size)
        self._vit_heads = int(vis.config.num_heads)
        self._vit_head_dim = self._vit_hidden // self._vit_heads
        self._vit_patch_w = vis.patch_embed.proj.weight.detach().to(device, torch.bfloat16).view(self._vit_hidden, -1).contiguous()
        self._vit_patch_b = vis.patch_embed.proj.bias.detach().to(device, torch.bfloat16).contiguous()
        self._vit_blocks = []
        for blk in vis.blocks:
            self._vit_blocks.append(
                {
                    "n1_w": blk.norm1.weight.detach().to(device, torch.bfloat16).contiguous(),
                    "n1_b": blk.norm1.bias.detach().to(device, torch.bfloat16).contiguous(),
                    "qkv_w": blk.attn.qkv.weight.detach().to(device, torch.bfloat16).contiguous(),
                    "qkv_b": blk.attn.qkv.bias.detach().to(device, torch.bfloat16).contiguous(),
                    "proj_w": blk.attn.proj.weight.detach().to(device, torch.bfloat16).contiguous(),
                    "proj_b": blk.attn.proj.bias.detach().to(device, torch.bfloat16).contiguous(),
                    "n2_w": blk.norm2.weight.detach().to(device, torch.bfloat16).contiguous(),
                    "n2_b": blk.norm2.bias.detach().to(device, torch.bfloat16).contiguous(),
                    "fc1_w": blk.mlp.linear_fc1.weight.detach().to(device, torch.bfloat16).contiguous(),
                    "fc1_b": blk.mlp.linear_fc1.bias.detach().to(device, torch.bfloat16).contiguous(),
                    "fc2_w": blk.mlp.linear_fc2.weight.detach().to(device, torch.bfloat16).contiguous(),
                    "fc2_b": blk.mlp.linear_fc2.bias.detach().to(device, torch.bfloat16).contiguous(),
                }
            )
        self._vit_merger = {
            "norm_w": vis.merger.norm.weight.detach().to(device, torch.bfloat16).contiguous(),
            "norm_b": vis.merger.norm.bias.detach().to(device, torch.bfloat16).contiguous(),
            "fc1_w": vis.merger.linear_fc1.weight.detach().to(device, torch.bfloat16).contiguous(),
            "fc1_b": vis.merger.linear_fc1.bias.detach().to(device, torch.bfloat16).contiguous(),
            "fc2_w": vis.merger.linear_fc2.weight.detach().to(device, torch.bfloat16).contiguous(),
            "fc2_b": vis.merger.linear_fc2.bias.detach().to(device, torch.bfloat16).contiguous(),
            "merge_hidden": int(vis.merger.hidden_size),
        }
        self._embed_tokens_w = vlm.get_input_embeddings().weight.detach().to(device, torch.bfloat16).contiguous()

        # ---- 2. Pack VLM LM layers (BF16 fused) ----
        self._vlm_layers = []
        full_idx = 0
        lin_idx = 0
        for layer_idx, layer in enumerate(lm.layers):
            ltype = self._layer_types[layer_idx]
            entry: dict[str, Any] = {
                "type": ltype,
                "in_norm_w1": (1.0 + layer.input_layernorm.weight.detach().float()).to(device, torch.bfloat16).contiguous(),
                "post_norm_w1": (1.0 + layer.post_attention_layernorm.weight.detach().float()).to(device, torch.bfloat16).contiguous(),
                "mlp_gu_w": torch.cat(
                    [layer.mlp.gate_proj.weight.detach(), layer.mlp.up_proj.weight.detach()], dim=0
                )
                .to(device, torch.bfloat16)
                .contiguous(),
                "mlp_down_w": layer.mlp.down_proj.weight.detach().to(device, torch.bfloat16).contiguous(),
            }
            if ltype == "linear_attention":
                la = layer.linear_attn
                entry["lin_idx"] = lin_idx
                lin_idx += 1
                entry["qkvzba_w"] = (
                    torch.cat(
                        [
                            la.in_proj_qkv.weight.detach(),
                            la.in_proj_z.weight.detach(),
                            la.in_proj_b.weight.detach(),
                            la.in_proj_a.weight.detach(),
                        ],
                        dim=0,
                    )
                    .to(device, torch.bfloat16)
                    .contiguous()
                )
                entry["conv_w"] = la.conv1d.weight.detach().squeeze(1).to(device, torch.bfloat16).contiguous()
                entry["neg_A_exp"] = (-la.A_log.detach().float().exp()).to(device).contiguous()
                entry["dt_bias"] = la.dt_bias.detach().float().to(device).contiguous()
                entry["gdn_norm_w"] = la.norm.weight.detach().to(device, torch.bfloat16).contiguous()
                entry["out_proj_w"] = la.out_proj.weight.detach().to(device, torch.bfloat16).contiguous()
            else:
                sa = layer.self_attn
                entry["full_idx"] = full_idx
                full_idx += 1
                entry["qkv_w"] = (
                    torch.cat(
                        [sa.q_proj.weight.detach(), sa.k_proj.weight.detach(), sa.v_proj.weight.detach()], dim=0
                    )
                    .to(device, torch.bfloat16)
                    .contiguous()
                )
                entry["q_norm_w1"] = (1.0 + sa.q_norm.weight.detach().float()).to(device, torch.bfloat16).contiguous()
                entry["k_norm_w1"] = (1.0 + sa.k_norm.weight.detach().float()).to(device, torch.bfloat16).contiguous()
                entry["o_proj_w"] = sa.o_proj.weight.detach().to(device, torch.bfloat16).contiguous()
            self._vlm_layers.append(entry)
        self._num_full_layers = full_idx
        self._num_lin_layers = lin_idx

        # ---- 3. Pack Action Expert layers (W8A16 for both LT prefill & 10x denoise) ----
        use_w8 = bool(getattr(self.config, "flashrt_w8a16_action_expert", True))
        chunk_m = int(self.config.chunk_size)
        lt_m = self.config.num_learnable_tokens + (0 if self.config.tokenize_state else 1)
        self._act_layers = []
        full_idx = 0
        lin_idx = 0
        for layer_idx, layer in enumerate(act.layers):
            ltype = self._layer_types[layer_idx]
            gu_w = torch.cat([layer.mlp.gate_proj.weight.detach(), layer.mlp.up_proj.weight.detach()], dim=0).to(
                device, torch.bfloat16
            )
            down_w = layer.mlp.down_proj.weight.detach().to(device, torch.bfloat16)
            entry = {
                "type": ltype,
                "in_norm_w1": (1.0 + layer.input_layernorm.weight.detach().float()).to(device, torch.bfloat16).contiguous(),
                "post_norm_w1": (1.0 + layer.post_attention_layernorm.weight.detach().float()).to(device, torch.bfloat16).contiguous(),
                "mlp_gu_den": _PackedLinear(gu_w, use_w8a16=use_w8, max_m=max(chunk_m, lt_m)),
                "mlp_down_den": _PackedLinear(down_w, use_w8a16=use_w8, max_m=max(chunk_m, lt_m)),
            }
            if ltype == "linear_attention":
                la = layer.linear_attn
                entry["lin_idx"] = lin_idx
                lin_idx += 1
                qkvz_w = torch.cat([la.in_proj_qkv.weight.detach(), la.in_proj_z.weight.detach()], dim=0).to(
                    device, torch.bfloat16
                )
                ba_w = torch.cat([la.in_proj_b.weight.detach(), la.in_proj_a.weight.detach()], dim=0).to(
                    device, torch.bfloat16
                )
                out_w = la.out_proj.weight.detach().to(device, torch.bfloat16)
                entry["qkvz_den"] = _PackedLinear(qkvz_w, use_w8a16=use_w8, max_m=max(chunk_m, lt_m))
                entry["ba_w"] = ba_w.contiguous()
                entry["conv_w"] = la.conv1d.weight.detach().squeeze(1).to(device, torch.bfloat16).contiguous()
                entry["neg_A_exp"] = (-la.A_log.detach().float().exp()).to(device).contiguous()
                entry["dt_bias"] = la.dt_bias.detach().float().to(device).contiguous()
                entry["gdn_norm_w"] = la.norm.weight.detach().to(device, torch.bfloat16).contiguous()
                entry["out_den"] = _PackedLinear(out_w, use_w8a16=use_w8, max_m=max(chunk_m, lt_m))
            else:
                sa = layer.self_attn
                entry["full_idx"] = full_idx
                full_idx += 1
                qkv_w = torch.cat(
                    [sa.q_proj.weight.detach(), sa.k_proj.weight.detach(), sa.v_proj.weight.detach()], dim=0
                ).to(device, torch.bfloat16)
                o_w = sa.o_proj.weight.detach().to(device, torch.bfloat16)
                entry["qkv_den"] = _PackedLinear(qkv_w, use_w8a16=use_w8, max_m=max(chunk_m, lt_m))
                entry["q_norm_w1"] = (1.0 + sa.q_norm.weight.detach().float()).to(device, torch.bfloat16).contiguous()
                entry["k_norm_w1"] = (1.0 + sa.k_norm.weight.detach().float()).to(device, torch.bfloat16).contiguous()
                entry["o_den"] = _PackedLinear(o_w, use_w8a16=use_w8, max_m=max(chunk_m, lt_m))
            self._act_layers.append(entry)

        self._act_final_norm_w1 = (1.0 + act.norm.weight.detach().float()).to(device, torch.bfloat16).contiguous()

        with torch.no_grad():
            self._lt_emb_const = (
                F.linear(
                    self.learnable_tokens.detach().to(device, torch.float32),
                    self.learnable_tokens_in_proj.weight.detach().to(device, torch.float32),
                    self.learnable_tokens_in_proj.bias.detach().to(device, torch.float32)
                    if self.learnable_tokens_in_proj.bias is not None
                    else None,
                )
                .to(torch.bfloat16)
                .contiguous()
            )

        self._act_in_w = self.action_in_proj.weight.detach().to(device, torch.bfloat16).contiguous()
        self._act_in_b = (
            self.action_in_proj.bias.detach().to(device, torch.bfloat16).contiguous()
            if self.action_in_proj.bias is not None
            else None
        )
        self._time_mlp_in_w = self.action_time_mlp_in.weight.detach().to(device, torch.bfloat16).contiguous()
        self._time_mlp_in_b = (
            self.action_time_mlp_in.bias.detach().to(device, torch.bfloat16).contiguous()
            if self.action_time_mlp_in.bias is not None
            else None
        )
        self._time_mlp_out_w = self.action_time_mlp_out.weight.detach().to(device, torch.bfloat16).contiguous()
        self._time_mlp_out_b = (
            self.action_time_mlp_out.bias.detach().to(device, torch.bfloat16).contiguous()
            if self.action_time_mlp_out.bias is not None
            else None
        )
        self._act_out_w_f32 = self.action_out_proj.weight.detach().to(device, torch.float32).contiguous()
        self._act_out_b_f32 = (
            self.action_out_proj.bias.detach().to(device, torch.float32).contiguous()
            if self.action_out_proj.bias is not None
            else None
        )
        if not self.config.tokenize_state:
            self._state_proj_w = self.state_proj.weight.detach().to(device, torch.bfloat16).contiguous()
            self._state_proj_b = (
                self.state_proj.bias.detach().to(device, torch.bfloat16).contiguous()
                if self.state_proj.bias is not None
                else None
            )

        self._packed = True
        logger.info(
            "Prepared InternVLA-A1.5 FlashRT backend (w8a16_action=%s, cuda_graph=%s, chunk=%d)",
            use_w8,
            getattr(self.config, "flashrt_use_cuda_graph", True),
            chunk_m,
        )

    def _get_vit_static(self, image_grid_thw: Tensor, device: torch.device) -> dict[str, Any]:
        grid_list = tuple(tuple(int(x) for x in row) for row in image_grid_thw.view(-1, 3).tolist())
        if grid_list in self._vit_static_cache:
            return self._vit_static_cache[grid_list]

        vis = self.qwen3_5_with_expert.qwen3_5.visual
        with torch.no_grad():
            pos_embeds = vis.fast_pos_embed_interpolate(image_grid_thw.view(-1, 3)).to(device, torch.bfloat16).contiguous()
            rotary = vis.rot_pos_emb(image_grid_thw.view(-1, 3)).to(device, torch.float32)
            emb = torch.cat((rotary, rotary), dim=-1)
            cos = emb.cos().to(torch.bfloat16).contiguous()
            sin = emb.sin().to(torch.bfloat16).contiguous()

        per_view = [t * h * w for t, h, w in grid_list]
        uniform = len(set(per_view)) == 1
        num_views = len(grid_list)
        patches_per_view = per_view[0]
        entry = {
            "grid_list": grid_list,
            "pos_embeds": pos_embeds,
            "cos": cos,
            "sin": sin,
            "uniform": uniform,
            "num_views": num_views,
            "patches_per_view": patches_per_view,
            "per_view": per_view,
        }
        self._vit_static_cache[grid_list] = entry
        return entry

    def _vit_forward(self, pipe: dict[str, Any], vit_static: dict[str, Any], stream: int) -> Tensor:
        ops = self._ops
        nv = vit_static["num_views"]
        p = vit_static["patches_per_view"]
        total_p = nv * p
        nh = self._vit_heads
        hd = self._vit_head_dim
        scale = hd**-0.5

        x = pipe["vit_x"]  # [total_p, vit_hidden]
        torch.addmm(self._vit_patch_b, pipe["pixel_values"], self._vit_patch_w.t(), out=x)
        x.add_(vit_static["pos_embeds"])

        vit_norm = pipe["vit_norm"]
        vit_qkv = pipe["vit_qkv"]
        vit_q = pipe["vit_q"]
        vit_k = pipe["vit_k"]
        vit_v = pipe["vit_v"]
        vit_proj_out = pipe["vit_proj_out"]
        vit_fc1_out = pipe["vit_fc1_out"]
        vit_fc2_out = pipe["vit_fc2_out"]
        cos_ptr = vit_static["cos"].data_ptr()
        sin_ptr = vit_static["sin"].data_ptr()

        if vit_static["uniform"] and hd == 64:
            # First block's pre-LN (no incoming residual add)
            b0 = self._vit_blocks[0]
            ops.residual_add_layernorm_bf16(
                x.data_ptr(), 0, b0["n1_w"].data_ptr(), b0["n1_b"].data_ptr(),
                vit_norm.data_ptr(), total_p, self._vit_hidden, 1e-6, stream,
            )
            for idx, blk in enumerate(self._vit_blocks):
                torch.addmm(blk["qkv_b"], vit_norm, blk["qkv_w"].t(), out=vit_qkv)
                ops.vit_qkv_rope_transpose_d64_bf16(
                    vit_qkv.data_ptr(), cos_ptr, sin_ptr,
                    vit_q.data_ptr(), vit_k.data_ptr(), vit_v.data_ptr(),
                    nv, p, nh, stream,
                )
                attn_out = F.scaled_dot_product_attention(vit_q, vit_k, vit_v, scale=scale)
                attn_flat = attn_out.transpose(1, 2).reshape(total_p, self._vit_hidden)
                torch.addmm(blk["proj_b"], attn_flat, blk["proj_w"].t(), out=vit_proj_out)

                ops.residual_add_layernorm_bf16(
                    x.data_ptr(), vit_proj_out.data_ptr(),
                    blk["n2_w"].data_ptr(), blk["n2_b"].data_ptr(),
                    vit_norm.data_ptr(), total_p, self._vit_hidden, 1e-6, stream,
                )
                torch.addmm(blk["fc1_b"], vit_norm, blk["fc1_w"].t(), out=vit_fc1_out)
                ops.gelu_tanh_inplace_bf16(vit_fc1_out.data_ptr(), total_p * self._vit_inter, stream)
                torch.addmm(blk["fc2_b"], vit_fc1_out, blk["fc2_w"].t(), out=vit_fc2_out)

                if idx + 1 < self._vit_depth:
                    bn = self._vit_blocks[idx + 1]
                    ops.residual_add_layernorm_bf16(
                        x.data_ptr(), vit_fc2_out.data_ptr(),
                        bn["n1_w"].data_ptr(), bn["n1_b"].data_ptr(),
                        vit_norm.data_ptr(), total_p, self._vit_hidden, 1e-6, stream,
                    )
                else:
                    # Fuse last residual add directly into Merger's LayerNorm!
                    m = self._vit_merger
                    ops.residual_add_layernorm_bf16(
                        x.data_ptr(), vit_fc2_out.data_ptr(),
                        m["norm_w"].data_ptr(), m["norm_b"].data_ptr(),
                        vit_norm.data_ptr(), total_p, self._vit_hidden, 1e-6, stream,
                    )
            m = self._vit_merger
            merged = vit_norm.view(-1, m["merge_hidden"])
            return F.linear(F.gelu(F.linear(merged, m["fc1_w"], m["fc1_b"])), m["fc2_w"], m["fc2_b"])

        # Fallback for non-uniform grids
        x_out = x
        per_view = vit_static["per_view"]
        cos_u = vit_static["cos"].unsqueeze(1)
        sin_u = vit_static["sin"].unsqueeze(1)
        for blk in self._vit_blocks:
            h = F.layer_norm(x_out, (self._vit_hidden,), blk["n1_w"], blk["n1_b"], 1e-6)
            qkv = F.linear(h, blk["qkv_w"], blk["qkv_b"]).view(total_p, 3, nh, hd)
            q0, k0, v0 = qkv[:, 0], qkv[:, 1], qkv[:, 2]
            q = q0 * cos_u + torch.cat((-q0[..., hd // 2 :], q0[..., : hd // 2]), dim=-1) * sin_u
            k = k0 * cos_u + torch.cat((-k0[..., hd // 2 :], k0[..., : hd // 2]), dim=-1) * sin_u
            qs = torch.split(q.transpose(0, 1).unsqueeze(0), per_view, dim=2)
            ks = torch.split(k.transpose(0, 1).unsqueeze(0), per_view, dim=2)
            vs = torch.split(v0.transpose(0, 1).unsqueeze(0), per_view, dim=2)
            outs = [F.scaled_dot_product_attention(qi, ki, vi, scale=scale) for qi, ki, vi in zip(qs, ks, vs)]
            attn_out = torch.cat(outs, dim=2).squeeze(0).transpose(0, 1).reshape(total_p, self._vit_hidden)
            x_out = x_out + F.linear(attn_out, blk["proj_w"], blk["proj_b"])
            h2 = F.layer_norm(x_out, (self._vit_hidden,), blk["n2_w"], blk["n2_b"], 1e-6)
            x_out = x_out + F.linear(F.gelu(F.linear(h2, blk["fc1_w"], blk["fc1_b"]), approximate="tanh"), blk["fc2_w"], blk["fc2_b"])
        m = self._vit_merger
        x_out = F.layer_norm(x_out, (self._vit_hidden,), m["norm_w"], m["norm_b"], 1e-6).view(-1, m["merge_hidden"])
        return F.linear(F.gelu(F.linear(x_out, m["fc1_w"], m["fc1_b"])), m["fc2_w"], m["fc2_b"])

    def _run_wy_gdn_fused(
        self,
        conv_out: Tensor,
        ba_in: Tensor,
        neg_A_exp: Tensor,
        dt_bias: Tensor,
        state_io: Tensor,
        wy: dict[str, Tensor],
        S: int,
        ba_stride: int,
        b_col_offset: int,
        stream: int,
    ) -> Tensor:
        """Fused gdn_prep_wy_bf16 + FlashRT 5-stage cuBLASLt Tensor-Core WY GDN."""
        fk = self._fk
        ops = self._ops
        H = self._lin_num_v_heads
        Hk = self._lin_num_k_heads
        D = self._lin_head_dim
        qk_g = self._lin_qk_group
        chunks = (S + 63) // 64

        q_l2 = wy["q_l2"][:S]
        k_l2 = wy["k_l2"][:S]
        v_buf = wy["v_buf"][:S]
        beta = wy["beta"][:S]
        g_cs = wy["g_cs"][:S]

        ops.gdn_prep_wy_bf16(
            conv_out.data_ptr(), ba_in.data_ptr(), neg_A_exp.data_ptr(), dt_bias.data_ptr(),
            q_l2.data_ptr(), k_l2.data_ptr(), v_buf.data_ptr(), beta.data_ptr(), g_cs.data_ptr(),
            S, H, D, ba_stride, b_col_offset, 1e-6, stream,
        )

        k_pack = wy["k_pack"][:chunks]
        kkt_base = wy["kkt_base"][:chunks]
        A = wy["A"][:chunks]
        Ai_pack = wy["Ai_pack"][:chunks]
        rhs_w = wy["rhs_w"][:chunks]
        rhs_u = wy["rhs_u"][:chunks]
        w_pack = wy["w_pack"][:chunks]
        u_pack = wy["u_pack"][:chunks]
        q_pack = wy["q_pack"][:chunks]
        h0 = wy["h0"][:chunks]
        out = wy["out"][:S]

        fk.linear_attn_gdn_wy_kkt_b64_bf16_cublaslt(
            k_l2.data_ptr(), beta.data_ptr(), g_cs.data_ptr(),
            k_pack.data_ptr(), kkt_base.data_ptr(), A.data_ptr(),
            S, Hk, H, D, qk_g, stream,
        )
        fk.linear_attn_gdn_wy_solve_tril_b64_f32_fused_pack_only(
            A.data_ptr(), Ai_pack.data_ptr(), S, H, stream,
        )
        fk.linear_attn_gdn_wy_recompute_wu_b64_bf16_cublaslt_packed_rhs(
            k_l2.data_ptr(), v_buf.data_ptr(), beta.data_ptr(), g_cs.data_ptr(),
            Ai_pack.data_ptr(), rhs_w.data_ptr(), rhs_u.data_ptr(),
            w_pack.data_ptr(), u_pack.data_ptr(), S, Hk, H, D, qk_g, stream,
        )
        fk.linear_attn_gdn_wy_chunk_h_b64_bf16_cublaslt_packed_wu(
            k_l2.data_ptr(), w_pack.data_ptr(), u_pack.data_ptr(), g_cs.data_ptr(),
            state_io.data_ptr(), h0.data_ptr(), 0,
            rhs_w.data_ptr(), rhs_u.data_ptr(), w_pack.data_ptr(),
            S, Hk, H, D, qk_g, stream,
        )
        fk.linear_attn_gdn_wy_output_o_b64_bf16_cublaslt_packed_kv(
            q_l2.data_ptr(), rhs_w.data_ptr(), rhs_u.data_ptr(), h0.data_ptr(), g_cs.data_ptr(),
            q_pack.data_ptr(), kkt_base.data_ptr(), Ai_pack.data_ptr(), w_pack.data_ptr(), u_pack.data_ptr(),
            out.data_ptr(), S, Hk, H, D, qk_g, stream,
        )
        return out

    def _alloc_wy_scratch(self, max_s: int, device: torch.device) -> dict[str, Tensor]:
        H = self._lin_num_v_heads
        Hk = self._lin_num_k_heads
        D = self._lin_head_dim
        chunks = (max_s + 63) // 64
        bf = torch.bfloat16
        f32 = torch.float32
        return {
            "q_l2": torch.empty(max_s, H, D, device=device, dtype=bf),
            "k_l2": torch.empty(max_s, Hk, D, device=device, dtype=bf),
            "v_buf": torch.empty(max_s, H, D, device=device, dtype=bf),
            "beta": torch.empty(max_s, H, device=device, dtype=bf),
            "g_cs": torch.empty(max_s, H, device=device, dtype=bf),
            "k_pack": torch.empty(chunks, Hk, 64, D, device=device, dtype=bf),
            "kkt_base": torch.empty(chunks, Hk, 64, 64, device=device, dtype=f32),
            "A": torch.empty(chunks, H, 64, 64, device=device, dtype=f32),
            "Ai_pack": torch.empty(chunks, H, 64, 64, device=device, dtype=bf),
            "rhs_w": torch.empty(chunks, H, 64, D, device=device, dtype=bf),
            "rhs_u": torch.empty(chunks, H, 64, D, device=device, dtype=bf),
            "w_pack": torch.empty(chunks, H, 64, D, device=device, dtype=bf),
            "u_pack": torch.empty(chunks, H, 64, D, device=device, dtype=bf),
            "q_pack": torch.empty(chunks, H, 64, D, device=device, dtype=bf),
            "h0": torch.empty(chunks, H, D, D, device=device, dtype=bf),
            "out": torch.empty(max_s, H, D, device=device, dtype=bf),
        }

    def _run_prefix_and_lt(self, pipe: dict[str, Any], vit_static: dict[str, Any], stream: int) -> None:
        """Graph 1 body: ViT -> Prefix Embeds -> VLM LM (24L, 128-aligned) -> 50-LT Action Expert (24L)."""
        fk = self._fk
        fa2 = self._fa2
        ops = self._ops
        L = pipe["prefix_len"]
        L_lt = pipe["lt_len"]
        eps = self._rms_eps
        H = self._lin_num_v_heads
        D = self._lin_head_dim
        vdim = H * D
        cdim = self._lin_conv_dim
        nh_q = self._num_q_heads
        nh_kv = self._num_kv_heads
        hd = self._head_dim
        rdim = self._rope_dim
        scale_fa = hd**-0.5
        qkv_full_stride = nh_q * 2 * hd + 2 * nh_kv * hd
        qkvzba_stride = cdim + vdim + 2 * H

        # 1. ViT + Prefix embeddings
        img_embs = self._vit_forward(pipe, vit_static, stream)
        x = pipe["vlm_x"]
        x.copy_(F.embedding(pipe["input_ids"][0], self._embed_tokens_w))
        x.index_copy_(0, pipe["img_pos_idx"], img_embs)

        # Compute RoPE cos/sin for prefix, LT, and action tokens inside graph
        lm_rot = self.qwen3_5_with_expert.qwen3_5.language_model.rotary_emb
        act_rot = self.qwen3_5_with_expert.action_expert.rotary_emb
        pref_cos, pref_sin = lm_rot(x.unsqueeze(0), pipe["prefix_pos_ids"])
        pref_cos = pref_cos[0].to(torch.bfloat16).contiguous()
        pref_sin = pref_sin[0].to(torch.bfloat16).contiguous()

        lt_pos = pipe["lt_pos_offsets"] + pipe["max_prefix_pos"]
        act_pos = pipe["act_pos_offsets"] + pipe["max_prefix_pos"]
        lt_cos, lt_sin = act_rot(pipe["lt_cos_dummy"], lt_pos)
        pipe["lt_cos"].copy_(lt_cos[0].to(torch.bfloat16))
        pipe["lt_sin"].copy_(lt_sin[0].to(torch.bfloat16))
        act_cos, act_sin = act_rot(pipe["act_cos_dummy"], act_pos)
        pipe["act_cos"].copy_(act_cos[0].to(torch.bfloat16))
        pipe["act_sin"].copy_(act_sin[0].to(torch.bfloat16))

        # 2. VLM LM 24 layers (fused, zero intermediate copies)
        vlm_norm = pipe["vlm_norm"]
        vlm_qkvzba = pipe["vlm_qkvzba"]
        vlm_qkv_in = pipe["vlm_qkv_in"]
        vlm_z_buf = pipe["vlm_z_buf"]
        vlm_conv_out = pipe["vlm_conv_out"]
        vlm_gdn_norm_out = pipe["vlm_gdn_norm_out"]
        vlm_full_qkv = pipe["vlm_full_qkv"]
        vlm_q_rot = pipe["vlm_q_rot"]
        vlm_attn_o = pipe["vlm_attn_o"]
        vlm_gated_o = pipe["vlm_gated_o"]
        vlm_lse = pipe["vlm_lse"]
        vlm_attn_out = pipe["vlm_attn_out"]
        vlm_mlp_gu = pipe["vlm_mlp_gu"]
        vlm_mlp_mid = pipe["vlm_mlp_mid"]
        vlm_mlp_down = pipe["vlm_mlp_down"]
        vlm_state = pipe["vlm_gdn_state"]
        wy_vlm = pipe["wy_vlm"]
        mask_bf = pipe["prefix_mask_bf16"]
        kv_k_list = pipe["kv_k"]
        kv_v_list = pipe["kv_v"]

        fk.rms_norm(x.data_ptr(), self._vlm_layers[0]["in_norm_w1"].data_ptr(), vlm_norm.data_ptr(), L, self._vlm_hidden, eps, stream)
        for idx, lw in enumerate(self._vlm_layers):
            if lw["type"] == "linear_attention":
                vlm_norm.mul_(mask_bf)
                torch.matmul(vlm_norm, lw["qkvzba_w"].t(), out=vlm_qkvzba)
                ops.split_qkvz_bf16(
                    vlm_qkvzba.data_ptr(), vlm_qkv_in.data_ptr(), vlm_z_buf.data_ptr(),
                    L, cdim, vdim, qkvzba_stride, stream,
                )
                fk.causal_conv1d_qwen36_bf16(
                    vlm_qkv_in.data_ptr(), lw["conv_w"].data_ptr(), 0, vlm_conv_out.data_ptr(),
                    1, L, cdim, 4, True, stream,
                )
                vlm_state.zero_()
                gdn_o = self._run_wy_gdn_fused(
                    vlm_conv_out, vlm_qkvzba, lw["neg_A_exp"], lw["dt_bias"],
                    vlm_state, wy_vlm, L, qkvzba_stride, cdim + vdim, stream,
                )
                fk.rms_norm_gated_silu_qwen36_bf16(
                    gdn_o.data_ptr(), vlm_z_buf.data_ptr(), lw["gdn_norm_w"].data_ptr(),
                    vlm_gdn_norm_out.data_ptr(), L * H, D, eps, stream,
                )
                torch.matmul(vlm_gdn_norm_out.view(L, vdim), lw["out_proj_w"].t(), out=vlm_attn_out)
            else:
                fidx = lw["full_idx"]
                torch.matmul(vlm_norm, lw["qkv_w"].t(), out=vlm_full_qkv)
                kv_k = kv_k_list[fidx]
                kv_v = kv_v_list[fidx]
                ops.qkv_split_norm_rope_kvwrite_bf16(
                    vlm_full_qkv.data_ptr(), lw["q_norm_w1"].data_ptr(), lw["k_norm_w1"].data_ptr(),
                    pref_cos.data_ptr(), pref_sin.data_ptr(), 0,
                    vlm_q_rot.data_ptr(), kv_k.data_ptr(), kv_v.data_ptr(),
                    L, nh_q, nh_kv, hd, rdim, eps, stream,
                )
                fa2.fwd_bf16_causal(
                    vlm_q_rot.data_ptr(), kv_k.data_ptr(), kv_v.data_ptr(),
                    vlm_attn_o.data_ptr(), vlm_lse.data_ptr(), 0, 0,
                    1, L, L, nh_q, nh_kv, hd,
                    (L * nh_q * hd, nh_q * hd, hd),
                    kv_k.stride()[:3], kv_v.stride()[:3],
                    (L * nh_q * hd, nh_q * hd, hd),
                    scale_fa, 0, stream,
                )
                ops.attn_gate_sigmoid_from_qkv_bf16(
                    vlm_attn_o.data_ptr(), vlm_full_qkv.data_ptr(), vlm_gated_o.data_ptr(),
                    L, nh_q, hd, qkv_full_stride, stream,
                )
                torch.matmul(vlm_gated_o, lw["o_proj_w"].t(), out=vlm_attn_out)

            fk.residual_add_rms_norm(
                x.data_ptr(), vlm_attn_out.data_ptr(), lw["post_norm_w1"].data_ptr(),
                vlm_norm.data_ptr(), L, self._vlm_hidden, eps, stream,
            )
            torch.matmul(vlm_norm, lw["mlp_gu_w"].t(), out=vlm_mlp_gu)
            ops.silu_mul_split_bf16(vlm_mlp_gu.data_ptr(), vlm_mlp_mid.data_ptr(), L, self._vlm_inter, stream)
            torch.matmul(vlm_mlp_mid, lw["mlp_down_w"].t(), out=vlm_mlp_down)
            if idx + 1 < self._num_layers:
                fk.residual_add_rms_norm(
                    x.data_ptr(), vlm_mlp_down.data_ptr(), self._vlm_layers[idx + 1]["in_norm_w1"].data_ptr(),
                    vlm_norm.data_ptr(), L, self._vlm_hidden, eps, stream,
                )
            else:
                x.add_(vlm_mlp_down)

        # 3. 50-Learnable-Token Action-Expert Prefill (ONCE per inference, W8A16 + fused kernels!)
        x_lt = pipe["lt_x_buf"]
        if self.config.tokenize_state:
            x_lt.copy_(self._lt_emb_const)
        else:
            st_emb = F.linear(pipe["state_bf16"], self._state_proj_w, self._state_proj_b)
            x_lt[0:1].copy_(st_emb)
            x_lt[1:].copy_(self._lt_emb_const)

        lt_norm = pipe["lt_norm"]
        lt_qkvz = pipe["lt_qkvz"]
        lt_qkv_in = pipe["lt_qkv_in"]
        lt_z_buf = pipe["lt_z_buf"]
        lt_ba = pipe["lt_ba"]
        lt_conv_out = pipe["lt_conv_out"]
        lt_gdn_norm_out = pipe["lt_gdn_norm_out"]
        lt_full_qkv = pipe["lt_full_qkv"]
        lt_q_rot = pipe["lt_q_rot"]
        lt_attn_o = pipe["lt_attn_o"]
        lt_gated_o = pipe["lt_gated_o"]
        lt_lse = pipe["lt_lse"]
        lt_attn_out = pipe["lt_attn_out"]
        lt_mlp_gu = pipe["lt_mlp_gu"]
        lt_mlp_mid = pipe["lt_mlp_mid"]
        lt_mlp_down = pipe["lt_mlp_down"]
        wy_lt = pipe["wy_lt"]
        lt_conv_states = pipe["lt_conv_states"]
        lt_gdn_states = pipe["lt_gdn_states"]
        lt_dst_idx = pipe["lt_dst_idx"]
        seqused_lt = pipe["seqused_lt"]
        max_kv_len = pipe["max_kv_len"]
        partials = pipe["w8_partials"]

        fk.rms_norm(x_lt.data_ptr(), self._act_layers[0]["in_norm_w1"].data_ptr(), lt_norm.data_ptr(), L_lt, self._act_hidden, eps, stream)
        for idx, lw in enumerate(self._act_layers):
            if lw["type"] == "linear_attention":
                lidx = lw["lin_idx"]
                lw["qkvz_den"].forward(fk, lt_norm, lt_qkvz, partials, stream)
                torch.matmul(lt_norm, lw["ba_w"].t(), out=lt_ba)
                ops.split_qkvz_bf16(
                    lt_qkvz.data_ptr(), lt_qkv_in.data_ptr(), lt_z_buf.data_ptr(),
                    L_lt, cdim, vdim, cdim + vdim, stream,
                )
                cs = lt_conv_states[lidx]
                cs.zero_()
                fk.causal_conv1d_qwen36_update_chunk_parallel_bf16(
                    lt_qkv_in.data_ptr(), lw["conv_w"].data_ptr(), 0,
                    lt_conv_out.data_ptr(), cs.data_ptr(),
                    1, L_lt, cdim, 4, True, stream,
                )
                gs = lt_gdn_states[lidx]
                gs.zero_()
                gdn_o = self._run_wy_gdn_fused(
                    lt_conv_out, lt_ba, lw["neg_A_exp"], lw["dt_bias"],
                    gs, wy_lt, L_lt, 2 * H, 0, stream,
                )
                fk.rms_norm_gated_silu_qwen36_bf16(
                    gdn_o.data_ptr(), lt_z_buf.data_ptr(), lw["gdn_norm_w"].data_ptr(),
                    lt_gdn_norm_out.data_ptr(), L_lt * H, D, eps, stream,
                )
                lw["out_den"].forward(fk, lt_gdn_norm_out.view(L_lt, vdim), lt_attn_out, partials, stream)
            else:
                fidx = lw["full_idx"]
                lw["qkv_den"].forward(fk, lt_norm, lt_full_qkv, partials, stream)
                kv_k = kv_k_list[fidx]
                kv_v = kv_v_list[fidx]
                ops.qkv_split_norm_rope_kvwrite_bf16(
                    lt_full_qkv.data_ptr(), lw["q_norm_w1"].data_ptr(), lw["k_norm_w1"].data_ptr(),
                    pipe["lt_cos"].data_ptr(), pipe["lt_sin"].data_ptr(), lt_dst_idx.data_ptr(),
                    lt_q_rot.data_ptr(), kv_k.data_ptr(), kv_v.data_ptr(),
                    L_lt, nh_q, nh_kv, hd, rdim, eps, stream,
                )
                fa2.fwd_bf16_seqused(
                    lt_q_rot.data_ptr(), kv_k.data_ptr(), kv_v.data_ptr(),
                    lt_attn_o.data_ptr(), lt_lse.data_ptr(), seqused_lt.data_ptr(),
                    1, L_lt, max_kv_len, nh_q, nh_kv, hd,
                    (L_lt * nh_q * hd, nh_q * hd, hd),
                    kv_k.stride()[:3], kv_v.stride()[:3],
                    (L_lt * nh_q * hd, nh_q * hd, hd),
                    scale_fa, 0, stream,
                )
                ops.attn_gate_sigmoid_from_qkv_bf16(
                    lt_attn_o.data_ptr(), lt_full_qkv.data_ptr(), lt_gated_o.data_ptr(),
                    L_lt, nh_q, hd, qkv_full_stride, stream,
                )
                lw["o_den"].forward(fk, lt_gated_o, lt_attn_out, partials, stream)

            fk.residual_add_rms_norm(
                x_lt.data_ptr(), lt_attn_out.data_ptr(), lw["post_norm_w1"].data_ptr(),
                lt_norm.data_ptr(), L_lt, self._act_hidden, eps, stream,
            )
            lw["mlp_gu_den"].forward(fk, lt_norm, lt_mlp_gu, partials, stream)
            ops.silu_mul_split_bf16(lt_mlp_gu.data_ptr(), lt_mlp_mid.data_ptr(), L_lt, self._act_inter, stream)
            lw["mlp_down_den"].forward(fk, lt_mlp_mid, lt_mlp_down, partials, stream)
            if idx + 1 < self._num_layers:
                fk.residual_add_rms_norm(
                    x_lt.data_ptr(), lt_mlp_down.data_ptr(), self._act_layers[idx + 1]["in_norm_w1"].data_ptr(),
                    lt_norm.data_ptr(), L_lt, self._act_hidden, eps, stream,
                )
            else:
                x_lt.add_(lt_mlp_down)

    def _run_denoise_loop(self, pipe: dict[str, Any], num_steps: int, stream: int) -> None:
        """Graph 2 body: All N flow-matching denoising steps unrolled with cached 50-LT state."""
        fk = self._fk
        fa2 = self._fa2
        ops = self._ops
        M = pipe["chunk_size"]
        eps = self._rms_eps
        H = self._lin_num_v_heads
        D = self._lin_head_dim
        vdim = H * D
        cdim = self._lin_conv_dim
        nh_q = self._num_q_heads
        nh_kv = self._num_kv_heads
        hd = self._head_dim
        rdim = self._rope_dim
        scale_fa = hd**-0.5
        qkv_full_stride = nh_q * 2 * hd + 2 * nh_kv * hd

        x_t = pipe["x_t"]
        x_t.copy_(pipe["noise"])
        dt = -1.0 / float(num_steps)

        act_norm = pipe["act_norm"]
        act_qkvz_buf = pipe["act_qkvz_buf"]
        act_qkv_in = pipe["act_qkv_in"]
        act_z_buf = pipe["act_z_buf"]
        act_ba = pipe["act_ba"]
        act_full_qkv_buf = pipe["act_full_qkv_buf"]
        act_attn_out_buf = pipe["act_attn_out_buf"]
        act_mlp_gu_buf = pipe["act_mlp_gu_buf"]
        act_mlp_down_buf = pipe["act_mlp_down_buf"]
        act_mlp_mid = pipe["act_mlp_mid"]
        act_conv_out = pipe["act_conv_out"]
        act_gdn_norm_out = pipe["act_gdn_norm_out"]
        act_q_rot = pipe["act_q_rot"]
        act_attn_o = pipe["act_attn_o"]
        act_gated_o = pipe["act_gated_o"]
        act_lse = pipe["act_lse"]
        wy_act = pipe["wy_act"]
        tmp_conv_state = pipe["tmp_conv_state"]
        tmp_gdn_state = pipe["tmp_gdn_state"]
        partials = pipe["w8_partials"]

        lt_conv_states = pipe["lt_conv_states"]
        lt_gdn_states = pipe["lt_gdn_states"]
        kv_k_list = pipe["kv_k"]
        kv_v_list = pipe["kv_v"]
        act_dst_idx = pipe["act_dst_idx"]
        seqused_act = pipe["seqused_act"]
        max_kv_len = pipe["max_kv_len"]
        step_time_embs = pipe["step_time_embs"]

        for s_idx in range(num_steps):
            act_emb = F.linear(x_t.to(torch.bfloat16), self._act_in_w, self._act_in_b)
            at_in = torch.cat([act_emb, step_time_embs[s_idx]], dim=-1)
            x = F.linear(F.silu(F.linear(at_in, self._time_mlp_in_w, self._time_mlp_in_b)), self._time_mlp_out_w, self._time_mlp_out_b)

            fk.rms_norm(x.data_ptr(), self._act_layers[0]["in_norm_w1"].data_ptr(), act_norm.data_ptr(), M, self._act_hidden, eps, stream)
            for idx, lw in enumerate(self._act_layers):
                if lw["type"] == "linear_attention":
                    lidx = lw["lin_idx"]
                    lw["qkvz_den"].forward(fk, act_norm, act_qkvz_buf, partials, stream)
                    torch.matmul(act_norm, lw["ba_w"].t(), out=act_ba)
                    ops.split_qkvz_bf16(
                        act_qkvz_buf.data_ptr(), act_qkv_in.data_ptr(), act_z_buf.data_ptr(),
                        M, cdim, vdim, cdim + vdim, stream,
                    )
                    tmp_conv_state.copy_(lt_conv_states[lidx])
                    fk.causal_conv1d_qwen36_update_chunk_parallel_bf16(
                        act_qkv_in.data_ptr(), lw["conv_w"].data_ptr(), 0,
                        act_conv_out.data_ptr(), tmp_conv_state.data_ptr(),
                        1, M, cdim, 4, True, stream,
                    )
                    tmp_gdn_state.copy_(lt_gdn_states[lidx])
                    gdn_o = self._run_wy_gdn_fused(
                        act_conv_out, act_ba, lw["neg_A_exp"], lw["dt_bias"],
                        tmp_gdn_state, wy_act, M, 2 * H, 0, stream,
                    )
                    fk.rms_norm_gated_silu_qwen36_bf16(
                        gdn_o.data_ptr(), act_z_buf.data_ptr(), lw["gdn_norm_w"].data_ptr(),
                        act_gdn_norm_out.data_ptr(), M * H, D, eps, stream,
                    )
                    lw["out_den"].forward(fk, act_gdn_norm_out.view(M, vdim), act_attn_out_buf, partials, stream)
                else:
                    fidx = lw["full_idx"]
                    lw["qkv_den"].forward(fk, act_norm, act_full_qkv_buf, partials, stream)
                    kv_k = kv_k_list[fidx]
                    kv_v = kv_v_list[fidx]
                    ops.qkv_split_norm_rope_kvwrite_bf16(
                        act_full_qkv_buf.data_ptr(), lw["q_norm_w1"].data_ptr(), lw["k_norm_w1"].data_ptr(),
                        pipe["act_cos"].data_ptr(), pipe["act_sin"].data_ptr(), act_dst_idx.data_ptr(),
                        act_q_rot.data_ptr(), kv_k.data_ptr(), kv_v.data_ptr(),
                        M, nh_q, nh_kv, hd, rdim, eps, stream,
                    )
                    fa2.fwd_bf16_seqused(
                        act_q_rot.data_ptr(), kv_k.data_ptr(), kv_v.data_ptr(),
                        act_attn_o.data_ptr(), act_lse.data_ptr(), seqused_act.data_ptr(),
                        1, M, max_kv_len, nh_q, nh_kv, hd,
                        (M * nh_q * hd, nh_q * hd, hd),
                        kv_k.stride()[:3], kv_v.stride()[:3],
                        (M * nh_q * hd, nh_q * hd, hd),
                        scale_fa, 0, stream,
                    )
                    ops.attn_gate_sigmoid_from_qkv_bf16(
                        act_attn_o.data_ptr(), act_full_qkv_buf.data_ptr(), act_gated_o.data_ptr(),
                        M, nh_q, hd, qkv_full_stride, stream,
                    )
                    lw["o_den"].forward(fk, act_gated_o, act_attn_out_buf, partials, stream)

                fk.residual_add_rms_norm(
                    x.data_ptr(), act_attn_out_buf.data_ptr(), lw["post_norm_w1"].data_ptr(),
                    act_norm.data_ptr(), M, self._act_hidden, eps, stream,
                )
                lw["mlp_gu_den"].forward(fk, act_norm, act_mlp_gu_buf, partials, stream)
                ops.silu_mul_split_bf16(act_mlp_gu_buf.data_ptr(), act_mlp_mid.data_ptr(), M, self._act_inter, stream)
                lw["mlp_down_den"].forward(fk, act_mlp_mid, act_mlp_down_buf, partials, stream)

                next_w = self._act_layers[idx + 1]["in_norm_w1"] if idx + 1 < self._num_layers else self._act_final_norm_w1
                fk.residual_add_rms_norm(
                    x.data_ptr(), act_mlp_down_buf.data_ptr(), next_w.data_ptr(),
                    act_norm.data_ptr(), M, self._act_hidden, eps, stream,
                )

            v_t = F.linear(act_norm.float(), self._act_out_w_f32, self._act_out_b_f32)
            x_t.add_(v_t, alpha=dt)

    def _get_or_build_pipeline(
        self,
        *,
        pv_rows: int,
        pv_cols: int,
        prefix_len: int,
        num_img_tokens: int,
        num_steps: int,
        vit_static: dict[str, Any],
        device: torch.device,
    ) -> dict[str, Any]:
        key = (pv_rows, pv_cols, prefix_len, num_img_tokens, num_steps)
        if key in self._pipelines:
            return self._pipelines[key]

        bf = torch.bfloat16
        f32 = torch.float32
        L = prefix_len
        L_lt = self.config.num_learnable_tokens + (0 if self.config.tokenize_state else 1)
        M = int(self.config.chunk_size)
        max_kv_len = L + L_lt + M
        H = self._lin_num_v_heads
        D = self._lin_head_dim
        vdim = H * D
        cdim = self._lin_conv_dim
        nh_q = self._num_q_heads
        nh_kv = self._num_kv_heads
        hd = self._head_dim
        rdim = self._rope_dim
        nv = vit_static["num_views"]
        ppv = vit_static["patches_per_view"]

        dt = -1.0 / float(num_steps)
        step_embs = []
        for s_idx in range(num_steps):
            t_val = torch.tensor([1.0 + s_idx * dt], dtype=f32, device=device)
            te = create_sinusoidal_pos_embedding(
                t_val, self._act_hidden, self.config.min_period, self.config.max_period, device=device
            ).to(bf)
            step_embs.append(te.expand(M, self._act_hidden).contiguous())
        step_time_embs = torch.stack(step_embs, dim=0).contiguous()

        pipe: dict[str, Any] = {
            "prefix_len": L,
            "lt_len": L_lt,
            "chunk_size": M,
            "max_kv_len": max_kv_len,
            "num_steps": num_steps,
            # Static inputs
            "pixel_values": torch.empty(pv_rows, pv_cols, device=device, dtype=bf),
            "input_ids": torch.empty(1, L, device=device, dtype=torch.long),
            "img_pos_idx": torch.empty(num_img_tokens, device=device, dtype=torch.long),
            "prefix_mask_bf16": torch.ones(L, 1, device=device, dtype=bf),
            "prefix_pos_ids": torch.zeros(3, 1, L, device=device, dtype=torch.long),
            "max_prefix_pos": torch.zeros(3, 1, 1, device=device, dtype=torch.long),
            "lt_pos_offsets": torch.arange(1, L_lt + 1, device=device, dtype=torch.long).view(1, 1, L_lt).expand(3, 1, L_lt).contiguous(),
            "act_pos_offsets": torch.arange(L_lt + 1, L_lt + M + 1, device=device, dtype=torch.long).view(1, 1, M).expand(3, 1, M).contiguous(),
            "lt_cos_dummy": torch.empty(1, L_lt, self._act_hidden, device=device, dtype=bf),
            "act_cos_dummy": torch.empty(1, M, self._act_hidden, device=device, dtype=bf),
            "lt_cos": torch.empty(L_lt, rdim, device=device, dtype=bf),
            "lt_sin": torch.empty(L_lt, rdim, device=device, dtype=bf),
            "act_cos": torch.empty(M, rdim, device=device, dtype=bf),
            "act_sin": torch.empty(M, rdim, device=device, dtype=bf),
            "state_bf16": torch.zeros(1, self.config.max_state_dim, device=device, dtype=bf),
            "lt_dst_idx": torch.arange(L, L + L_lt, device=device, dtype=torch.long),
            "act_dst_idx": torch.arange(L + L_lt, L + L_lt + M, device=device, dtype=torch.long),
            "seqused_lt": torch.tensor([L + L_lt], device=device, dtype=torch.int32),
            "seqused_act": torch.tensor([L + L_lt + M], device=device, dtype=torch.int32),
            "noise": torch.zeros(M, self.config.max_action_dim, device=device, dtype=f32),
            "x_t": torch.zeros(M, self.config.max_action_dim, device=device, dtype=f32),
            "step_time_embs": step_time_embs,
            # ViT scratch
            "vit_x": torch.empty(pv_rows, self._vit_hidden, device=device, dtype=bf),
            "vit_norm": torch.empty(pv_rows, self._vit_hidden, device=device, dtype=bf),
            "vit_qkv": torch.empty(pv_rows, 3 * self._vit_hidden, device=device, dtype=bf),
            "vit_q": torch.empty(nv, self._vit_heads, ppv, self._vit_head_dim, device=device, dtype=bf),
            "vit_k": torch.empty(nv, self._vit_heads, ppv, self._vit_head_dim, device=device, dtype=bf),
            "vit_v": torch.empty(nv, self._vit_heads, ppv, self._vit_head_dim, device=device, dtype=bf),
            "vit_proj_out": torch.empty(pv_rows, self._vit_hidden, device=device, dtype=bf),
            "vit_fc1_out": torch.empty(pv_rows, self._vit_inter, device=device, dtype=bf),
            "vit_fc2_out": torch.empty(pv_rows, self._vit_hidden, device=device, dtype=bf),
            # Shared KV cache across VLM + LT + Action Expert (6 full-attn layers)
            "kv_k": [torch.empty(1, max_kv_len, nh_kv, hd, device=device, dtype=bf) for _ in range(self._num_full_layers)],
            "kv_v": [torch.empty(1, max_kv_len, nh_kv, hd, device=device, dtype=bf) for _ in range(self._num_full_layers)],
            # VLM scratch
            "vlm_x": torch.empty(L, self._vlm_hidden, device=device, dtype=bf),
            "vlm_norm": torch.empty(L, self._vlm_hidden, device=device, dtype=bf),
            "vlm_qkvzba": torch.empty(L, cdim + vdim + 2 * H, device=device, dtype=bf),
            "vlm_qkv_in": torch.empty(L, cdim, device=device, dtype=bf),
            "vlm_z_buf": torch.empty(L, vdim, device=device, dtype=bf),
            "vlm_conv_out": torch.empty(L, cdim, device=device, dtype=bf),
            "vlm_gdn_norm_out": torch.empty(L * H, D, device=device, dtype=bf),
            "vlm_full_qkv": torch.empty(L, nh_q * hd * 2 + 2 * nh_kv * hd, device=device, dtype=bf),
            "vlm_q_rot": torch.empty(L, nh_q, hd, device=device, dtype=bf),
            "vlm_attn_o": torch.empty(L, nh_q, hd, device=device, dtype=bf),
            "vlm_gated_o": torch.empty(L, nh_q * hd, device=device, dtype=bf),
            "vlm_lse": torch.empty(1, nh_q, L, device=device, dtype=f32),
            "vlm_attn_out": torch.empty(L, self._vlm_hidden, device=device, dtype=bf),
            "vlm_mlp_gu": torch.empty(L, self._vlm_inter * 2, device=device, dtype=bf),
            "vlm_mlp_mid": torch.empty(L, self._vlm_inter, device=device, dtype=bf),
            "vlm_mlp_down": torch.empty(L, self._vlm_hidden, device=device, dtype=bf),
            "vlm_gdn_state": torch.zeros(H, D, D, device=device, dtype=bf),
            "wy_vlm": self._alloc_wy_scratch(L, device),
            # LT prefill scratch + cached states (18 linear-attn layers)
            "lt_x_buf": torch.empty(L_lt, self._act_hidden, device=device, dtype=bf),
            "lt_norm": torch.empty(L_lt, self._act_hidden, device=device, dtype=bf),
            "lt_qkvz": torch.empty(L_lt, cdim + vdim, device=device, dtype=bf),
            "lt_qkv_in": torch.empty(L_lt, cdim, device=device, dtype=bf),
            "lt_z_buf": torch.empty(L_lt, vdim, device=device, dtype=bf),
            "lt_ba": torch.empty(L_lt, 2 * H, device=device, dtype=bf),
            "lt_conv_out": torch.empty(L_lt, cdim, device=device, dtype=bf),
            "lt_gdn_norm_out": torch.empty(L_lt * H, D, device=device, dtype=bf),
            "lt_full_qkv": torch.empty(L_lt, nh_q * hd * 2 + 2 * nh_kv * hd, device=device, dtype=bf),
            "lt_q_rot": torch.empty(L_lt, nh_q, hd, device=device, dtype=bf),
            "lt_attn_o": torch.empty(L_lt, nh_q, hd, device=device, dtype=bf),
            "lt_gated_o": torch.empty(L_lt, nh_q * hd, device=device, dtype=bf),
            "lt_lse": torch.empty(1, nh_q, L_lt, device=device, dtype=f32),
            "lt_attn_out": torch.empty(L_lt, self._act_hidden, device=device, dtype=bf),
            "lt_mlp_gu": torch.empty(L_lt, self._act_inter * 2, device=device, dtype=bf),
            "lt_mlp_mid": torch.empty(L_lt, self._act_inter, device=device, dtype=bf),
            "lt_mlp_down": torch.empty(L_lt, self._act_hidden, device=device, dtype=bf),
            "wy_lt": self._alloc_wy_scratch(L_lt, device),
            "lt_conv_states": [torch.zeros(1, 3, cdim, device=device, dtype=bf) for _ in range(self._num_lin_layers)],
            "lt_gdn_states": [torch.zeros(H, D, D, device=device, dtype=bf) for _ in range(self._num_lin_layers)],
            # Action Expert denoise scratch
            "act_norm": torch.empty(M, self._act_hidden, device=device, dtype=bf),
            "act_qkvz_buf": torch.empty(M, cdim + vdim, device=device, dtype=bf),
            "act_qkv_in": torch.empty(M, cdim, device=device, dtype=bf),
            "act_z_buf": torch.empty(M, vdim, device=device, dtype=bf),
            "act_ba": torch.empty(M, 2 * H, device=device, dtype=bf),
            "act_full_qkv_buf": torch.empty(M, nh_q * hd * 2 + 2 * nh_kv * hd, device=device, dtype=bf),
            "act_attn_out_buf": torch.empty(M, self._act_hidden, device=device, dtype=bf),
            "act_mlp_gu_buf": torch.empty(M, self._act_inter * 2, device=device, dtype=bf),
            "act_mlp_down_buf": torch.empty(M, self._act_hidden, device=device, dtype=bf),
            "act_mlp_mid": torch.empty(M, self._act_inter, device=device, dtype=bf),
            "act_conv_out": torch.empty(M, cdim, device=device, dtype=bf),
            "act_gdn_norm_out": torch.empty(M * H, D, device=device, dtype=bf),
            "act_q_rot": torch.empty(M, nh_q, hd, device=device, dtype=bf),
            "act_attn_o": torch.empty(M, nh_q, hd, device=device, dtype=bf),
            "act_gated_o": torch.empty(M, nh_q * hd, device=device, dtype=bf),
            "act_lse": torch.empty(1, nh_q, M, device=device, dtype=f32),
            "wy_act": self._alloc_wy_scratch(M, device),
            "tmp_conv_state": torch.zeros(1, 3, cdim, device=device, dtype=bf),
            "tmp_gdn_state": torch.zeros(H, D, D, device=device, dtype=bf),
            "w8_partials": torch.empty(8, max(M, L_lt), max(cdim + vdim, self._act_inter * 2), device=device, dtype=f32),
            "graph_prefix": None,
            "graph_denoise": None,
            "vit_static": vit_static,
        }

        self._pipelines[key] = pipe
        return pipe

    def _capture_graphs(self, pipe: dict[str, Any], vit_static: dict[str, Any], num_steps: int) -> None:
        cap_stream = torch.cuda.Stream()
        torch.cuda.synchronize()

        with torch.amp.autocast(device_type="cuda", enabled=False):
            with torch.cuda.stream(cap_stream):
                s_ptr = int(cap_stream.cuda_stream)
                for _ in range(2):
                    self._run_prefix_and_lt(pipe, vit_static, s_ptr)
                    self._run_denoise_loop(pipe, num_steps, s_ptr)
            cap_stream.synchronize()

            g1 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g1, stream=cap_stream):
                self._run_prefix_and_lt(pipe, vit_static, int(cap_stream.cuda_stream))

            g2 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g2, stream=cap_stream):
                self._run_denoise_loop(pipe, num_steps, int(cap_stream.cuda_stream))

        torch.cuda.synchronize()
        pipe["graph_prefix"] = g1
        pipe["graph_denoise"] = g2
        logger.info(
            "Captured InternVLA-A1.5 FlashRT CUDA Graphs (prefix_len=%d, lt=%d, chunk=%d, steps=%d)",
            pipe["prefix_len"],
            pipe["lt_len"],
            pipe["chunk_size"],
            num_steps,
        )

    def _get_cached_position_ids(
        self,
        lang_tokens_eff: Tensor,
        thw: Tensor,
        pad_masks_eff: Tensor,
        grid_list: tuple[tuple[int, int, int], ...],
        valid_len: int,
        img_first: int,
        img_last: int,
    ) -> tuple[Tensor, Tensor]:
        key = (int(lang_tokens_eff.shape[1]), valid_len, img_first, img_last, grid_list)
        cached = self._pos_ids_cache.get(key)
        if cached is not None:
            return cached
        pos_ids, _ = self.get_position_ids(lang_tokens_eff, thw, pad_masks_eff)
        max_pos = pos_ids.max(dim=-1, keepdim=True).values
        self._pos_ids_cache[key] = (pos_ids, max_pos)
        return pos_ids, max_pos

    @torch.no_grad()
    def sample_actions(
        self,
        pixel_values,
        image_grid_thw,
        lang_tokens,
        lang_masks,
        state,
        fast_token_mask: Tensor | None = None,
        noise=None,
        num_steps=None,
    ) -> Tensor:
        if not state.is_cuda:
            raise RuntimeError("InternVLAA15FlashRT requires CUDA.")
        if num_steps is None:
            num_steps = int(self.config.num_inference_steps)
        else:
            num_steps = int(num_steps)

        if int(state.shape[0]) != 1:
            return super().sample_actions(
                pixel_values, image_grid_thw, lang_tokens, lang_masks, state,
                fast_token_mask=fast_token_mask, noise=noise, num_steps=num_steps,
            )

        if (
            self.config.block_action_attend_fast_tokens
            and fast_token_mask is not None
            and bool(fast_token_mask.any().item())
        ):
            return super().sample_actions(
                pixel_values, image_grid_thw, lang_tokens, lang_masks, state,
                fast_token_mask=fast_token_mask, noise=noise, num_steps=num_steps,
            )

        device = state.device
        if not self._packed:
            self.prepare_flashrt(device)

        pv = pixel_values.view(-1, pixel_values.shape[-1])
        thw = image_grid_thw.view(-1, 3)
        vit_static = self._get_vit_static(thw, device)

        pad_masks_full = lang_masks.to(torch.bool)
        raw_len = int(lang_tokens.shape[1])
        valid_len = int(pad_masks_full[0].sum().item())

        # Align VLM sequence length to a multiple of 128 (SM87 cuBLAS Tensor-Core wave alignment!)
        # For right-padded prompts with valid_len <= 384 (e.g. 380 in a 416 buffer), slicing to 384
        # saves ~43 ms on VLM MLP GEMMs + 1 WY chunk with bit-exact semantics.
        aligned_len = max(128, ((valid_len + 127) // 128) * 128)
        if aligned_len <= raw_len and bool(pad_masks_full[0, :valid_len].all().item()):
            lang_tokens_eff = lang_tokens[:, :aligned_len]
            pad_masks_eff = pad_masks_full[:, :aligned_len]
            prefix_len = aligned_len
        elif aligned_len > raw_len:
            pad_extra = aligned_len - raw_len
            lang_tokens_eff = F.pad(lang_tokens, (0, pad_extra), value=0)
            pad_masks_eff = F.pad(pad_masks_full, (0, pad_extra), value=False)
            prefix_len = aligned_len
        else:
            lang_tokens_eff = lang_tokens
            pad_masks_eff = pad_masks_full
            prefix_len = raw_len

        img_pos = (lang_tokens_eff[0] == self._image_token_id).nonzero(as_tuple=True)[0]
        num_img_tokens = int(img_pos.numel())
        img_first = int(img_pos[0].item()) if num_img_tokens > 0 else -1
        img_last = int(img_pos[-1].item()) if num_img_tokens > 0 else -1

        pipe = self._get_or_build_pipeline(
            pv_rows=int(pv.shape[0]),
            pv_cols=int(pv.shape[1]),
            prefix_len=prefix_len,
            num_img_tokens=num_img_tokens,
            num_steps=num_steps,
            vit_static=vit_static,
            device=device,
        )

        pipe["pixel_values"].copy_(pv.to(torch.bfloat16))
        pipe["input_ids"].copy_(lang_tokens_eff)
        pipe["img_pos_idx"].copy_(img_pos)
        pipe["prefix_mask_bf16"].copy_(pad_masks_eff[0, :, None].to(torch.bfloat16))

        L_lt = pipe["lt_len"]
        M = pipe["chunk_size"]
        pipe["seqused_lt"].fill_(valid_len + L_lt)
        pipe["seqused_act"].fill_(valid_len + L_lt + M)
        pipe["lt_dst_idx"].copy_(torch.arange(valid_len, valid_len + L_lt, device=device, dtype=torch.long))
        pipe["act_dst_idx"].copy_(torch.arange(valid_len + L_lt, valid_len + L_lt + M, device=device, dtype=torch.long))

        prefix_pos_ids, max_pos = self._get_cached_position_ids(
            lang_tokens_eff, thw, pad_masks_eff, vit_static["grid_list"], valid_len, img_first, img_last
        )
        pipe["prefix_pos_ids"].copy_(prefix_pos_ids)
        pipe["max_prefix_pos"].copy_(max_pos)

        if not self.config.tokenize_state:
            pipe["state_bf16"].copy_(state.to(torch.bfloat16))

        if noise is None:
            pipe["noise"].normal_(0.0, 1.0)
        else:
            pipe["noise"].copy_(noise[0].float())

        use_cg = bool(getattr(self.config, "flashrt_use_cuda_graph", True))
        if use_cg:
            if pipe["graph_prefix"] is None:
                self._capture_graphs(pipe, vit_static, num_steps)
            pipe["graph_prefix"].replay()
            pipe["graph_denoise"].replay()
        else:
            s_ptr = int(torch.cuda.current_stream(device).cuda_stream)
            with torch.amp.autocast(device_type="cuda", enabled=False):
                self._run_prefix_and_lt(pipe, vit_static, s_ptr)
                self._run_denoise_loop(pipe, num_steps, s_ptr)

        return pipe["x_t"].unsqueeze(0).clone()
