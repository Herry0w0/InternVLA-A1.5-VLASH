#!/usr/bin/env python3
"""Benchmark InternVLA-A1.5 inference latency (3-view vs 4-view) for Jetson Orin / GPU.

Measures end-to-end `predict_action_chunk` wall-clock latency with `action_loss_only=True`
and `inference_backend="optimized"` (zero WAN2.2 overhead, SDPA + CUDA Graph replay),
and prints recommended VLASH `vlash_min_delay_steps` / `vlash_max_delay_steps` at target FPS.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-path", type=Path, default=Path("/mnt/data3/InternVLA-A1.5-base"))
    p.add_argument("--vlm-path", type=Path, default=Path("/mnt/data3/Qwen3.5-2B"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--backend", choices=["optimized", "standard"], default="optimized")
    p.add_argument("--num-views", type=int, default=3, help="Number of camera views (3 recommended for Orin).")
    p.add_argument("--chunk-size", type=int, default=30)
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import InternVLAA15Policy
    from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (
        InternVLAA15ChatProcessorTransformFn,
    )
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    device = torch.device(args.device)
    cfg = PreTrainedConfig.from_pretrained(args.ckpt_path)
    cfg.vlm_model_name_or_path = str(args.vlm_path)
    cfg.chunk_size = int(args.chunk_size)
    cfg.n_action_steps = int(args.chunk_size)
    cfg.num_inference_steps = int(args.num_inference_steps)
    cfg.action_loss_only = True
    cfg.inference_backend = args.backend
    cfg.device = str(device)

    policy = InternVLAA15Policy.from_pretrained(args.ckpt_path, config=cfg)
    policy.to(device)
    policy.eval()

    eval_pad_length = (416 if args.num_views <= 3 else 512) if args.backend == "optimized" else None
    processor = InternVLAA15ChatProcessorTransformFn(
        pretrained_model_name_or_path=str(args.vlm_path),
        max_length=650 if args.num_views <= 3 else 800,
        num_views=args.num_views,
        tokenize_state=cfg.tokenize_state,
        max_state_dim=cfg.max_state_dim,
        norm_mode="mean_std",
        vlash_condition_on_delay=True,
        eval_output_tag="<Action>",
        eval_pad_length=eval_pad_length,
        use_fast_action_tokens=False,
        mode="eval",
        action_mode="joint",
    )

    sample = {
        OBS_STATE: torch.zeros(16, dtype=torch.float32),
        "task": "Flip the parcel so the label faces up",
        "vlash_delay_steps": torch.tensor(6, dtype=torch.long),
    }
    for i in range(args.num_views):
        sample[f"{OBS_IMAGES}.image{i}"] = torch.rand(3, 224, 224, dtype=torch.float32)
        sample[f"{OBS_IMAGES}.image{i}_mask"] = True

    processed = processor(sample)
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32
    batch = {}
    for k, v in processed.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k in ("observation.pixel_values", "observation.image_grid_thw"):
            batch[k] = v.to(device)
        elif v.dtype in (torch.int64, torch.int32, torch.bool):
            batch[k] = v.unsqueeze(0).to(device)
        else:
            batch[k] = v.unsqueeze(0).to(device=device, dtype=dtype)

    with torch.no_grad():
        for _ in range(args.warmup):
            out = policy.predict_action_chunk(batch)
            if not torch.isfinite(out).all():
                raise RuntimeError("Non-finite action output detected during warmup!")
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        latencies_ms = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            out = policy.predict_action_chunk(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            if not torch.isfinite(out).all():
                raise RuntimeError("Non-finite action output detected during benchmark!")

    arr = np.asarray(latencies_ms, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    p99 = float(np.percentile(arr, 99))
    mean_ms = float(arr.mean())
    rec_center = int(round((p50 / 1000.0) * args.fps))
    rec_min = max(1, rec_center - 2)
    rec_max = max(rec_min + 2, int(math.ceil((p99 / 1000.0) * args.fps)) + 2)

    print("=== InternVLA-A1.5 Inference Latency Benchmark ===")
    print(f"Backend          : {args.backend} (action_loss_only=True)")
    print(f"Views / Chunk    : {args.num_views} views / chunk_size={args.chunk_size} / steps={args.num_inference_steps}")
    print(f"Mean / P50       : {mean_ms:.1f} ms / {p50:.1f} ms ({1000.0 / p50:.2f} Hz)")
    print(f"P90 / P99        : {p90:.1f} ms / {p99:.1f} ms")
    print(f"Recommended VLASH @ {args.fps:.0f}Hz:")
    print(f"  vlash_min_delay_steps : {rec_min}")
    print(f"  vlash_max_delay_steps : {rec_max}")
    print(f"  wan_video_span_steps  : {args.chunk_size + rec_max}")


if __name__ == "__main__":
    main()
