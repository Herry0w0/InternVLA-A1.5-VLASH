#!/usr/bin/env python3
"""Recompute exact float64 relative-action and state quantile statistics for InternVLA-A1.5 + VLASH.

Note: `convert_mcap_to_internvla_lerobot_v3.py` already computes VLASH relative-action stats in one shot
(and supports `--step finalize,verify --force` to recompute stats in-place). Keep this helper for
recomputing stats on an existing LeRobot v3.0 dataset directory with optional hardlink copy.

Copies a LeRobot v3.0 dataset via hardlinks (zero extra disk usage for .parquet/.mp4)
and recomputes global float64 stats (min, max, mean, std, q01, q10, q50, q90, q99, count) for:
  - observation.state (16-dim absolute joint + gripper states)
  - action (chunk_size=30 relative actions: action[t:t+chunk_size] - state[t] on arm joints,
    keeping grippers [indices 7, 15] absolute; excludes masked padded tail by default to match
    `mask_action_pad: true`)

Includes automatic stationary-joint quantile/std widening so q01_q99 or mean_std normalization
never explodes on nearly-constant joints.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from lerobot.dataset_schemas import get_schema
from lerobot.datasets.compute_stats import RunningQuantileStats
from lerobot.datasets.lerobot_dataset import LeRobotDataset

QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]
QUANTILE_KEYS = [f"q{int(q * 100):02d}" for q in QUANTILES]
EXACT_QUANTILE_ROW_LIMIT = 5_000_000


def copy_with_hardlinks(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            copy_with_hardlinks(item, target)
        elif item.is_symlink():
            target.symlink_to(os.readlink(item))
        elif item.suffix in (".parquet", ".mp4", ".avi", ".mov", ".webm"):
            try:
                os.link(item, target)
            except OSError:
                shutil.copy2(item, target)
        else:
            shutil.copy2(item, target)


def compute_exact_vector_stats(arr: np.ndarray) -> dict[str, np.ndarray]:
    """Compute exact float64 statistics on [N, D]."""
    arr = np.asarray(arr, dtype=np.float64)
    q_vals = np.quantile(arr, QUANTILES, axis=0)
    stats: dict[str, np.ndarray] = {
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "mean": np.mean(arr, axis=0),
        "std": np.std(arr, axis=0),
        "count": np.array([arr.shape[0]], dtype=np.int64),
    }
    for k, qv in zip(QUANTILE_KEYS, q_vals, strict=True):
        stats[k] = qv
    return stats


def compute_episode_rel_actions(
    states: np.ndarray,
    actions: np.ndarray,
    *,
    chunk_size: int,
    action_mask: np.ndarray,
    include_padded_tail: bool = False,
) -> np.ndarray:
    ep_len = states.shape[0]
    t_idx = np.arange(ep_len)[:, None]
    k_idx = np.arange(chunk_size)[None, :]
    step_idx = t_idx + k_idx
    clamped_idx = np.minimum(step_idx, ep_len - 1)
    ref = np.where(action_mask, states, 0.0)[:, None, :]
    rel_chunks = actions[clamped_idx] - ref
    if include_padded_tail:
        return rel_chunks.reshape(-1, actions.shape[1])
    return rel_chunks[step_idx < ep_len]


def protect_stationary_dims(
    stats: dict[str, np.ndarray],
    action_mask: np.ndarray,
    arm_min_range: float = 0.1,
    arm_min_std: float = 0.02,
) -> dict[str, np.ndarray]:
    """Widen near-constant dimensions so q99-q01 and std never blow up normalization."""
    out = {k: v.copy() for k, v in stats.items()}
    dim = out["mean"].shape[0]
    for d in range(dim):
        is_arm = bool(action_mask[d])
        if is_arm:
            min_range = arm_min_range
            min_std = arm_min_std
        else:
            scale_span = float(max(abs(out["max"][d]), abs(out["min"][d])))
            if scale_span > 2.0:
                min_range = 10.0
                min_std = 2.0
            else:
                min_range = 0.1
                min_std = 0.02

        cur_range = float(out["q99"][d] - out["q01"][d])
        if cur_range < min_range:
            center = float(out["mean"][d])
            half = min_range / 2.0
            out["q01"][d] = min(float(out["q01"][d]), center - half)
            out["q99"][d] = max(float(out["q99"][d]), center + half)
            if float(out["q99"][d] - out["q01"][d]) < min_range:
                out["q01"][d] = center - half
                out["q99"][d] = center + half
            logging.info(
                "Widened dim %d (is_arm=%s) quantile range %.5f -> [%.4f, %.4f]",
                d,
                is_arm,
                cur_range,
                out["q01"][d],
                out["q99"][d],
            )
        if float(out["std"][d]) < min_std:
            out["std"][d] = min_std
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/mnt/data3/jiangxirui/dynamic_rigid_parcel_flip_internvla_lerobot_v3"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/mnt/data3/jiangxirui/dynamic_rigid_parcel_flip_internvla_lerobot_v3_rel30"),
    )
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--arm-min-range", type=float, default=0.1)
    parser.add_argument("--arm-min-std", type=float, default=0.02)
    parser.add_argument(
        "--include-padded-tail-in-stats",
        action="store_true",
        help="Include clamped episode-tail actions in relative stats (off by default to match mask_action_pad=true).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.output_root.resolve() != args.source_root.resolve():
        if args.output_root.exists():
            shutil.rmtree(args.output_root)
        logging.info("Hardlink-copying %s -> %s", args.source_root, args.output_root)
        copy_with_hardlinks(args.source_root, args.output_root)

    ds = LeRobotDataset(repo_id=".", root=args.output_root)
    schema = get_schema(ds.meta.robot_type)
    state_dim = ds.meta.features["observation.state"]["shape"][0]
    action_mask = (
        schema.action_mask.numpy()
        if schema.action_mask.numel() == state_dim
        else np.ones(state_dim, dtype=bool)
    )

    from_ids = np.asarray(ds.meta.episodes["dataset_from_index"])
    to_ids = np.asarray(ds.meta.episodes["dataset_to_index"])

    total_rel_rows = 0
    for from_idx, to_idx in zip(from_ids, to_ids, strict=True):
        ep_len = int(to_idx - from_idx)
        if args.include_padded_tail_in_stats:
            total_rel_rows += ep_len * args.chunk_size
        else:
            full_cnt = max(0, ep_len - args.chunk_size)
            tail_n = min(ep_len, args.chunk_size)
            total_rel_rows += full_cnt * args.chunk_size + (tail_n * (tail_n + 1)) // 2

    use_streaming = total_rel_rows > EXACT_QUANTILE_ROW_LIMIT
    rel_runner = RunningQuantileStats() if use_streaming else None
    all_states: list[np.ndarray] = []
    all_rel_actions: list[np.ndarray] = []

    for from_idx, to_idx in zip(from_ids, to_ids, strict=True):
        ep = ds.hf_dataset.select(np.arange(from_idx, to_idx))
        states = torch.stack(ep["observation.state"][:]).numpy().astype(np.float64)
        actions = torch.stack(ep["action"][:]).numpy().astype(np.float64)
        all_states.append(states)

        rel_ep = compute_episode_rel_actions(
            states,
            actions,
            chunk_size=args.chunk_size,
            action_mask=action_mask,
            include_padded_tail=args.include_padded_tail_in_stats,
        )
        if rel_runner is not None:
            rel_runner.update(rel_ep)
        else:
            all_rel_actions.append(rel_ep)

    states_cat = np.concatenate(all_states, axis=0)
    raw_rel_stats = (
        rel_runner.get_statistics()
        if rel_runner is not None
        else compute_exact_vector_stats(np.concatenate(all_rel_actions, axis=0))
    )

    state_stats = protect_stationary_dims(
        compute_exact_vector_stats(states_cat),
        action_mask=action_mask,
        arm_min_range=args.arm_min_range,
        arm_min_std=args.arm_min_std,
    )
    action_stats = protect_stationary_dims(
        raw_rel_stats,
        action_mask=action_mask,
        arm_min_range=args.arm_min_range,
        arm_min_std=args.arm_min_std,
    )

    stats_path = args.output_root / "meta" / "stats.json"
    bak_path = args.output_root / "meta" / "stats.json.abs_bak"
    if stats_path.exists() and not bak_path.exists():
        shutil.copy2(stats_path, bak_path)

    with stats_path.open("r", encoding="utf-8") as f:
        stats_json = json.load(f)

    stats_json["observation.state"] = {k: v.tolist() for k, v in state_stats.items()}
    stats_json["action"] = {k: v.tolist() for k, v in action_stats.items()}

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats_json, f, ensure_ascii=False, indent=4)

    logging.info(
        "Recomputed relative stats (chunk_size=%d, include_padded_tail=%s, mask=%s) -> %s",
        args.chunk_size,
        args.include_padded_tail_in_stats,
        action_mask.tolist(),
        stats_path,
    )


if __name__ == "__main__":
    main()
