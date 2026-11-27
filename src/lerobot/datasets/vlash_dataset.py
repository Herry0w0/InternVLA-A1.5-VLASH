"""VLASH (Future-State-Aware Asynchronous Inference) & Optional Dual-Frame History Dataset Wrapper."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Callable

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames


def detect_letterbox_borders(
    tensor_img: torch.Tensor, threshold: float = 0.04
) -> tuple[int, int, int, int]:
    """Detect contiguous near-black letterbox border widths (top, bottom, left, right).

    Works on ``[C, H, W]`` or ``[N, C, H, W]`` float tensors in ``[0, 1]``.
    Uses a small threshold (``0.04`` ~ 10/255) to accommodate H.264 YUV420p border ringing.
    Returns ``(0, 0, 0, 0)`` if the frame appears completely dark to avoid wiping real frames.
    """
    if tensor_img.ndim == 3:
        img = tensor_img.unsqueeze(0)
    elif tensor_img.ndim == 4:
        img = tensor_img
    else:
        return (0, 0, 0, 0)

    h, w = int(img.shape[-2]), int(img.shape[-1])
    # Max across batch and channel dims -> [H, W]
    max_hw = img.amax(dim=(0, 1))
    row_max = max_hw.amax(dim=1)  # [H]
    col_max = max_hw.amax(dim=0)  # [W]

    nonblack_rows = (row_max > threshold).nonzero(as_tuple=False).flatten()
    nonblack_cols = (col_max > threshold).nonzero(as_tuple=False).flatten()
    if nonblack_rows.numel() == 0 or nonblack_cols.numel() == 0:
        return (0, 0, 0, 0)

    top = int(nonblack_rows[0].item())
    bottom = h - 1 - int(nonblack_rows[-1].item())
    left = int(nonblack_cols[0].item())
    right = w - 1 - int(nonblack_cols[-1].item())

    # Guard against accidentally cropping >45% from any edge on very dark scenes
    if top + bottom >= int(0.9 * h):
        top, bottom = 0, 0
    if left + right >= int(0.9 * w):
        left, right = 0, 0
    return (top, bottom, left, right)


class VLASHLeRobotDataset(LeRobotDataset):
    """LeRobotDataset subclass implementing VLASH temporal-offset sampling and fixed-span WAN foresight.

    At sample index ``idx`` (observation time ``t``):
      - Sample random delay ``delta`` (capped to remaining frames in episode so ``abs_exec < ep_end``):
          * ``0`` with probability ``vlash_zero_delay_prob`` (or when at episode end),
          * else uniform integer in ``[vlash_min_delay_steps, min(vlash_max_delay_steps, remaining)]``.
      - ``idx_exec = abs_obs + delta`` is the future handover step (effective ``delta = idx_exec - abs_obs``
        is stored in ``item["vlash_delay_steps"]`` for explicit prompt conditioning).
      - Visual policy inputs are read at observation time ``t``:
          * Standard 3-camera setup (recommended for Jetson Orin latency): ``head_right``, ``left_arm``,
            ``right_arm`` at ``t``.
          * Optional head history (``enable_head_history=True``): also loads ``t - head_history_gap``
            stored under ``f"{head_camera_key}_prev"``.
      - Proprioceptive ``observation.state`` is read at ``t + delta`` (``idx_exec``).
      - Target ``action`` chunk ``[t + delta, ..., t + delta + chunk_size - 1]`` and
        ``action_is_pad`` are read starting at ``idx_exec``.
      - Clean (un-augmented) WAN2.2 future video frames are sampled uniformly across a **fixed**
        horizon ``[t, t + wan_video_span_steps]`` (invariant to ``delta``) so Foresight Tokens always
        predict consistent physical dynamics without variable-speed time warping.
      - Letterbox-safe photometric augmentation preserves exact zero letterbox borders per camera.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        *,
        vlash_enabled: bool = True,
        vlash_min_delay_steps: int = 1,
        vlash_max_delay_steps: int = 12,
        vlash_zero_delay_prob: float = 0.1,
        wan_video_span_steps: int | None = None,
        enable_head_history: bool = False,
        head_history_gap: int = 4,
        head_camera_key: str = "observation.images.head_right",
        chunk_size: int = 30,
        num_video_frames: int = 4,
        letterbox_pad_px: int = -1,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            revision=revision,
            force_cache_sync=force_cache_sync,
            download_videos=download_videos,
            video_backend=video_backend,
            batch_encoding_size=batch_encoding_size,
        )
        self.vlash_enabled = bool(vlash_enabled)
        self.vlash_max_delay_steps = max(0, int(vlash_max_delay_steps))
        self.vlash_min_delay_steps = (
            min(max(1, int(vlash_min_delay_steps)), self.vlash_max_delay_steps)
            if self.vlash_max_delay_steps > 0
            else 0
        )
        self.vlash_zero_delay_prob = float(vlash_zero_delay_prob)
        self.enable_head_history = bool(enable_head_history)
        self.head_history_gap = int(head_history_gap)
        self.head_camera_key = str(head_camera_key)
        self.head_prev_key = f"{self.head_camera_key}_prev"
        self.chunk_size = int(chunk_size)
        self.num_video_frames = int(num_video_frames)
        # Fixed WAN foresight span (independent of random per-sample delay)
        if wan_video_span_steps is not None and int(wan_video_span_steps) > 0:
            self.wan_video_span_steps = int(wan_video_span_steps)
        else:
            self.wan_video_span_steps = self.chunk_size + (
                self.vlash_max_delay_steps if self.vlash_enabled else 0
            )
        self.letterbox_pad_px = int(letterbox_pad_px)

    def _sample_delay_steps(self, max_allowed_delay: int | None = None) -> int:
        if not self.vlash_enabled or self.vlash_max_delay_steps <= 0:
            return 0
        hi = self.vlash_max_delay_steps
        if max_allowed_delay is not None:
            hi = min(hi, max(0, int(max_allowed_delay)))
        if hi <= 0:
            return 0
        if random.random() < self.vlash_zero_delay_prob:
            return 0
        lo = min(self.vlash_min_delay_steps, hi)
        return random.randint(lo, hi)

    def _apply_letterbox_safe_transform(
        self, tensor_img: torch.Tensor
    ) -> torch.Tensor:
        if self.image_transforms is None:
            return tensor_img

        if self.letterbox_pad_px < 0:
            top, bottom, left, right = detect_letterbox_borders(tensor_img)
        elif self.letterbox_pad_px > 0:
            pad = self.letterbox_pad_px
            h = int(tensor_img.shape[-2])
            top, bottom = (pad, pad) if 2 * pad < h else (0, 0)
            left, right = 0, 0
        else:
            top, bottom, left, right = 0, 0, 0, 0

        out = self.image_transforms(tensor_img)
        if top > 0 or bottom > 0 or left > 0 or right > 0:
            out = out.clone()
            h, w = int(out.shape[-2]), int(out.shape[-1])
            if top > 0:
                out[..., :top, :] = 0.0
            if bottom > 0:
                out[..., h - bottom :, :] = 0.0
            if left > 0:
                out[..., :, :left] = 0.0
            if right > 0:
                out[..., :, w - right :] = 0.0
        return out.clamp(0.0, 1.0)

    def _to_relative_index(self, abs_idx: int) -> int:
        if self._absolute_to_relative_idx is None:
            return abs_idx
        return self._absolute_to_relative_idx[abs_idx]

    def _get_timestamp_at(self, abs_idx: int) -> float:
        rel_idx = self._to_relative_index(abs_idx)
        return float(self.hf_dataset[rel_idx]["timestamp"].item())

    def _decode_frames_for_indices(
        self, vid_key: str, ep_idx: int, frame_indices: list[int]
    ) -> torch.Tensor:
        """Decode deduplicated timestamps for ``frame_indices`` -> [len, C, H, W]."""
        ep = self.meta.episodes[ep_idx]
        from_ts = float(ep[f"videos/{vid_key}/from_timestamp"])
        video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)

        unique_indices: list[int] = []
        pos_map: dict[int, int] = {}
        for fi in frame_indices:
            if fi not in pos_map:
                pos_map[fi] = len(unique_indices)
                unique_indices.append(fi)

        shifted_ts = [from_ts + self._get_timestamp_at(fi) for fi in unique_indices]
        try:
            decoded = decode_video_frames(
                video_path, shifted_ts, self.tolerance_s, self.video_backend
            )
            if decoded.ndim == 3:
                decoded = decoded.unsqueeze(0)
        except Exception as exc:
            logging.error(
                "[vlash_video_decode_error] repo=%s ep=%d key=%s ts=%s err=%r",
                self.repo_id,
                ep_idx,
                vid_key,
                shifted_ts,
                exc,
            )
            h, w, c = self.meta.features[vid_key]["shape"]
            decoded = torch.zeros((len(unique_indices), c, h, w), dtype=torch.float32)

        gather = [pos_map[fi] for fi in frame_indices]
        return decoded[gather]

    def __getitem__(self, idx: int) -> dict:
        self._ensure_hf_dataset_loaded()
        item = self.hf_dataset[idx]
        ep_idx = int(item["episode_index"].item())
        ep = self.meta.episodes[ep_idx]
        ep_start = int(ep["dataset_from_index"])
        ep_end = int(ep["dataset_to_index"])

        # HF index when episodes filter is active uses global index column
        abs_obs = int(item["index"].item()) if "index" in item else idx
        remaining = max(0, (ep_end - 1) - abs_obs)
        delay = self._sample_delay_steps(max_allowed_delay=remaining)
        abs_exec = min(ep_end - 1, abs_obs + delay)
        effective_delay = abs_exec - abs_obs
        abs_prev = max(ep_start, abs_obs - self.head_history_gap)

        # State at t + delta (idx_exec)
        rel_exec = self._to_relative_index(abs_exec)
        exec_row = self.hf_dataset[rel_exec]
        if "observation.state" in exec_row:
            item["observation.state"] = exec_row["observation.state"]

        # Action chunk [t + delta, ..., t + delta + chunk_size - 1]
        action_indices = [
            min(ep_end - 1, abs_exec + k) for k in range(self.chunk_size)
        ]
        action_is_pad = torch.BoolTensor(
            [(abs_exec + k >= ep_end) for k in range(self.chunk_size)]
        )
        rel_action_indices = [self._to_relative_index(ai) for ai in action_indices]
        try:
            action_chunk = torch.stack(self.hf_dataset["action"][rel_action_indices])
        except (KeyError, TypeError, IndexError):
            action_chunk = torch.stack(self.hf_dataset[rel_action_indices]["action"])
        item["action"] = action_chunk
        item["action_is_pad"] = action_is_pad
        item["vlash_delay_steps"] = torch.tensor(effective_delay, dtype=torch.long)

        # Decode videos
        if len(self.meta.video_keys) > 0:
            # Fixed-horizon 5-frame WAN span across [t, t + wan_video_span_steps]
            span = self.wan_video_span_steps
            wan_indices = [
                min(ep_end - 1, abs_obs + (span * i) // self.num_video_frames)
                for i in range(self.num_video_frames + 1)
            ]

            for vid_key in self.meta.video_keys:
                if vid_key == self.head_camera_key:
                    query_list = [abs_obs]
                    if self.enable_head_history:
                        query_list.append(abs_prev)
                    query_list.extend(wan_indices)

                    decoded = self._decode_frames_for_indices(vid_key, ep_idx, query_list)
                    head_curr = decoded[0]
                    offset = 1
                    if self.enable_head_history:
                        head_prev = decoded[1]
                        offset = 2
                    wan_frames = decoded[offset:]  # clean [T, C, H, W] in [0, 1]
                    item["observation.video_frames_raw"] = wan_frames

                    if self.enable_head_history:
                        # Temporally consistent augmentation on stacked [head_prev, head_curr]
                        pair = self._apply_letterbox_safe_transform(
                            torch.stack([head_prev, head_curr], dim=0)
                        )
                        item[self.head_prev_key] = pair[0]
                        item[vid_key] = pair[1]
                    else:
                        item[vid_key] = self._apply_letterbox_safe_transform(head_curr)
                else:
                    decoded = self._decode_frames_for_indices(vid_key, ep_idx, [abs_obs])
                    item[vid_key] = self._apply_letterbox_safe_transform(decoded[0])

        # Task string and robot_type
        if "galaxea" in self.repo_id and "coarse_task_index" in item:
            task_idx = int(item["coarse_task_index"].item())
        else:
            task_idx = int(item["task_index"].item())
        item["task"] = self.meta.tasks.iloc[task_idx].name
        item["robot_type"] = self.meta.robot_type
        return item
