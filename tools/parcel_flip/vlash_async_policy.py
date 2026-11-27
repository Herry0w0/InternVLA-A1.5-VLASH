"""VLASH (Future-State-Aware Asynchronous Inference) Runtime for InternVLA-A1.5.

Implements:
  1. Future-state roll-forward (VLASH, arXiv:2512.01031) with optional servo tracking-lag
     compensation between commanded targets (`singorix/wbcs/target`) and measured sensors
     (`singorix/wbcs/sensor`).
  2. Explicit delay prompt conditioning (`; Delay: <delta>`) & exact training-time prompt
     parity (`; Output: <Action>`, `mean_std` or `q01_q99` normalization, 224x224 letterbox).
  3. Zero-WAN optimized CUDA-Graph inference (`action_loss_only=True`,
     `inference_backend="optimized"`).
  4. Optional VLASH Action Quantization (`action_quant_ratio >= 1`) and Galbot SDK
     right-arm-first / left-arm-first adapter.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_ARM_ACTION_MASK = np.array(
    [True] * 7 + [False] + [True] * 7 + [False], dtype=bool
)


def swap_dual_arm_order(vec: np.ndarray) -> np.ndarray:
    """Swap [right_arm(7), right_grip(1), left_arm(7), left_grip(1)] <-> [left(8), right(8)]."""
    arr = np.asarray(vec)
    if arr.shape[-1] != 16:
        raise ValueError(f"Expected 16-dim dual-arm vector, got shape {arr.shape}")
    return np.concatenate([arr[..., 8:16], arr[..., 0:8]], axis=-1)


def unnormalize_vector(
    x: np.ndarray,
    stats: dict[str, Any],
    mode: str = "mean_std",
    eps: float = 1e-6,
) -> np.ndarray:
    """Invert NormalizeTransformFn for mean_std / min_max / q01_q99."""
    arr = np.asarray(x, dtype=np.float32)
    if mode == "mean_std":
        mean = np.asarray(stats["mean"], dtype=np.float32)
        std = np.asarray(stats["std"], dtype=np.float32)
        return arr * (std + eps) + mean
    if mode == "min_max":
        min_v = np.asarray(stats["min"], dtype=np.float32)
        max_v = np.asarray(stats["max"], dtype=np.float32)
        return (arr + 1.0) / 2.0 * (max_v - min_v + eps) + min_v
    if mode == "q01_q99":
        min_v = np.asarray(stats["q01"], dtype=np.float32)
        max_v = np.asarray(stats["q99"], dtype=np.float32)
        return (arr + 1.0) / 2.0 * (max_v - min_v + eps) + min_v
    raise ValueError(f"Unsupported norm_mode: {mode!r}")


def reconstruct_absolute_actions(
    raw_chunk: np.ndarray,
    ref_state: np.ndarray,
    action_mode: str = "delta",
    action_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Convert unnormalized model output chunk [T, D] into absolute joint + gripper targets."""
    chunk = np.asarray(raw_chunk, dtype=np.float32).copy()
    if action_mode != "delta":
        return chunk
    ref = np.asarray(ref_state, dtype=np.float32)
    mask = DEFAULT_ARM_ACTION_MASK if action_mask is None else np.asarray(action_mask, dtype=bool)
    if mask.shape[0] != chunk.shape[-1]:
        raise ValueError(f"action_mask length {mask.shape[0]} != action dim {chunk.shape[-1]}")
    offset = np.where(mask, ref, 0.0)[None, :]
    chunk += offset
    return chunk


def quantize_action_chunk(chunk: np.ndarray, quant_ratio: int = 1) -> np.ndarray:
    """VLASH Action Quantization: group consecutive micro-actions by stride `quant_ratio`."""
    q = int(quant_ratio)
    if q <= 1:
        return chunk
    # Select the endpoint of each q-step macro-action group (preserving final target if uneven)
    indices = list(range(q - 1, chunk.shape[0], q))
    if not indices or indices[-1] != chunk.shape[0] - 1:
        indices.append(chunk.shape[0] - 1)
    return chunk[indices]


@dataclass
class VLASHRollForwardResult:
    """Result of planning an asynchronous VLASH inference call."""

    future_state: np.ndarray
    delay_steps: int
    trigger_step: int
    handover_step: int


class VLASHFutureStateController:
    """Pure-Python/NumPy controller for VLASH future-state roll-forward & async chunk handover.

    Maintains:
      - Current global control step counter ``step``
      - Committed action buffer ``_action_queue`` (each entry is a 16-dim absolute target)
      - Exponential moving average of inference latency ``ema_infer_s``
    """

    def __init__(
        self,
        *,
        fps: float = 30.0,
        chunk_size: int = 30,
        min_delay_steps: int = 0,
        max_delay_steps: int = 12,
        fixed_delay_steps: int | None = None,
        servo_lag_steps: int = 0,
        action_quant_ratio: int = 1,
        latency_ema_alpha: float = 0.3,
        initial_latency_s: float = 0.20,
        min_remaining_to_trigger: int | None = None,
    ) -> None:
        self.fps = float(fps)
        self.chunk_size = int(chunk_size)
        self.min_delay_steps = max(0, int(min_delay_steps))
        self.max_delay_steps = max(0, int(max_delay_steps))
        self.fixed_delay_steps = fixed_delay_steps
        self.servo_lag_steps = max(0, int(servo_lag_steps))
        self.action_quant_ratio = max(1, int(action_quant_ratio))
        self.latency_ema_alpha = float(latency_ema_alpha)
        self.ema_infer_s = float(initial_latency_s)
        # Trigger next async inference early enough that remaining committed steps >= delay + safety
        self.min_remaining_to_trigger = min_remaining_to_trigger

        self.step: int = 0
        self._action_queue: deque[np.ndarray] = deque()
        self._pending_plan: VLASHRollForwardResult | None = None
        self._pending_chunk: np.ndarray | None = None

    def reset(self) -> None:
        self.step = 0
        self._action_queue.clear()
        self._pending_plan = None
        self._pending_chunk = None

    def update_latency(self, infer_s: float) -> None:
        infer_s = max(0.0, float(infer_s))
        a = self.latency_ema_alpha
        self.ema_infer_s = a * infer_s + (1.0 - a) * self.ema_infer_s

    def estimate_delay_steps(self) -> int:
        if self.fixed_delay_steps is not None:
            return max(0, min(self.max_delay_steps, int(self.fixed_delay_steps)))
        est = int(round(self.ema_infer_s * self.fps))
        return max(self.min_delay_steps, min(self.max_delay_steps, est))

    def plan_future_state(self, current_sensor_state: np.ndarray) -> VLASHRollForwardResult:
        """Roll forward ``current_sensor_state`` by ``delta`` committed actions.

        When the committed queue has fewer than ``delta`` actions (e.g., cold start at step 0),
        clamps ``delta`` to ``len(_action_queue)`` (0 on cold start -> synchronous first chunk).
        If ``servo_lag_steps > 0``, looks slightly earlier in the committed target queue to match
        the physical sensor tracking lag between commanded targets and measured joint positions.
        """
        sensor = np.asarray(current_sensor_state, dtype=np.float32).copy()
        desired_delay = self.estimate_delay_steps()
        avail = len(self._action_queue)

        if avail == 0 or desired_delay == 0:
            plan = VLASHRollForwardResult(
                future_state=sensor,
                delay_steps=0,
                trigger_step=self.step,
                handover_step=self.step,
            )
            self._pending_plan = plan
            return plan

        effective_delay = min(desired_delay, avail)
        # Index in committed queue whose execution brings robot to t + effective_delay.
        # Subtract servo_lag_steps when targets lead measured sensors by ~1-2 frames.
        queue_idx = effective_delay - 1 - self.servo_lag_steps
        if queue_idx < 0:
            future_state = sensor
        else:
            future_state = np.asarray(self._action_queue[queue_idx], dtype=np.float32).copy()

        plan = VLASHRollForwardResult(
            future_state=future_state,
            delay_steps=effective_delay,
            trigger_step=self.step,
            handover_step=self.step + effective_delay,
        )
        self._pending_plan = plan
        return plan

    def should_trigger_inference(self) -> bool:
        """Return True when the controller should launch a new background inference."""
        if self._pending_plan is not None:
            return False
        est_delay = self.estimate_delay_steps()
        if self.min_remaining_to_trigger is not None:
            threshold = self.min_remaining_to_trigger
        else:
            # Leave enough committed actions to cover inference + 2-step jitter margin
            threshold = max(est_delay + 2, self.chunk_size // 2)
        return len(self._action_queue) <= threshold

    def commit_predicted_chunk(
        self,
        absolute_chunk: np.ndarray,
        plan: VLASHRollForwardResult,
    ) -> None:
        """Stage a newly predicted absolute chunk [T, D] for handover at ``plan.handover_step``."""
        chunk = quantize_action_chunk(
            np.asarray(absolute_chunk, dtype=np.float32),
            quant_ratio=self.action_quant_ratio,
        )
        self._pending_chunk = chunk
        self._pending_plan = plan
        # Apply immediately if we have already reached or passed the handover step
        self._maybe_handover()

    def _maybe_handover(self) -> None:
        if self._pending_chunk is None or self._pending_plan is None:
            return
        if self.step < self._pending_plan.handover_step:
            return
        overdue = self.step - self._pending_plan.handover_step
        if overdue >= len(self._pending_chunk):
            # Inference took longer than the entire predicted chunk; keep last target
            remaining = self._pending_chunk[-1:]
        else:
            remaining = self._pending_chunk[overdue:]
        self._action_queue = deque(remaining[i].copy() for i in range(len(remaining)))
        self._pending_chunk = None
        self._pending_plan = None

    def pop_action(self) -> np.ndarray | None:
        """Pop the next absolute action for the current control step and advance step counter."""
        self._maybe_handover()
        if not self._action_queue:
            return None
        action = self._action_queue.popleft()
        self.step += 1
        return action

    @property
    def queue_length(self) -> int:
        return len(self._action_queue)


class VLASHAsyncInternVLAPolicy:
    """End-to-end asynchronous VLASH inference wrapper for InternVLA-A1.5 checkpoints.

    Automatically reads ``train_config.json`` and ``stats.json`` saved alongside the checkpoint
    so normalization mode (``mean_std`` / ``q01_q99``), action mode (``delta`` / ``abs``),
    prompt format (``; Delay: <d>`` and ``; Output: <Action>``), and camera setup match training.
    """

    def __init__(
        self,
        *,
        policy: Any,
        processor: Any,
        state_normalizer: Any,
        action_stats: dict[str, Any],
        norm_mode: str,
        action_mode: str,
        action_mask: np.ndarray,
        chunk_size: int,
        action_dim: int,
        device: str,
        dtype: torch.dtype,
        fps: float = 30.0,
        min_delay_steps: int = 0,
        max_delay_steps: int = 12,
        fixed_delay_steps: int | None = None,
        servo_lag_steps: int = 0,
        action_quant_ratio: int = 1,
        enable_head_history: bool = False,
        head_history_gap: int = 4,
        sdk_right_arm_first: bool = False,
        default_task: str = "Flip the parcel so the label faces up",
    ) -> None:
        self.policy = policy
        self.processor = processor
        self.state_normalizer = state_normalizer
        self.action_stats = action_stats
        self.norm_mode = norm_mode
        self.action_mode = action_mode
        self.action_mask = action_mask
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.dtype = dtype
        self.enable_head_history = bool(enable_head_history)
        self.head_history_gap = int(head_history_gap)
        self.sdk_right_arm_first = bool(sdk_right_arm_first)
        self.default_task = default_task

        self.controller = VLASHFutureStateController(
            fps=fps,
            chunk_size=chunk_size,
            min_delay_steps=min_delay_steps,
            max_delay_steps=max_delay_steps,
            fixed_delay_steps=fixed_delay_steps,
            servo_lag_steps=servo_lag_steps,
            action_quant_ratio=action_quant_ratio,
        )
        self._head_history: deque[torch.Tensor] = deque(maxlen=max(1, head_history_gap + 1))
        self._lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._last_error: Exception | None = None

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir: str | Path,
        *,
        vlm_model_path: str | None = None,
        device: str = "cuda",
        inference_backend: str = "optimized",
        fps: float = 30.0,
        fixed_delay_steps: int | None = None,
        servo_lag_steps: int = 0,
        action_quant_ratio: int = 1,
        sdk_right_arm_first: bool = False,
        default_task: str = "Flip the parcel so the label faces up",
    ) -> VLASHAsyncInternVLAPolicy:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.dataset_schemas import get_schema
        from lerobot.datasets.utils import cast_stats_to_numpy, load_json
        from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import InternVLAA15Policy
        from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (
            InternVLAA15ChatProcessorTransformFn,
        )
        from lerobot.transforms.core import NormalizeTransformFn
        from lerobot.utils.constants import ACTION, OBS_STATE

        ckpt_path = Path(ckpt_dir)
        cfg = PreTrainedConfig.from_pretrained(ckpt_path)
        if vlm_model_path:
            cfg.vlm_model_name_or_path = vlm_model_path
        cfg.action_loss_only = True
        cfg.inference_backend = inference_backend
        cfg.device = device

        policy = InternVLAA15Policy.from_pretrained(ckpt_path, config=cfg)
        policy.to(device)
        policy.eval()

        # Load training config metadata for exact preprocessing parity
        train_cfg_path = ckpt_path / "train_config.json"
        ds_cfg: dict[str, Any] = {}
        if train_cfg_path.is_file():
            with train_cfg_path.open("r", encoding="utf-8") as f:
                train_cfg = json.load(f)
            ds_cfg = train_cfg.get("dataset", {})

        norm_mode = str(ds_cfg.get("norm_mode", "mean_std"))
        action_mode = str(ds_cfg.get("action_mode", "delta"))
        vlash_enabled = bool(ds_cfg.get("vlash_enabled", True))
        vlash_condition_on_delay = bool(ds_cfg.get("vlash_condition_on_delay", True)) and vlash_enabled
        vlash_max_delay = int(ds_cfg.get("vlash_max_delay_steps", 12))
        enable_head_history = bool(ds_cfg.get("enable_head_history", False))
        head_history_gap = int(ds_cfg.get("head_history_gap", 4))
        max_prompt_length = int(ds_cfg.get("max_prompt_length", 650))
        eval_output_tag = str(ds_cfg.get("eval_output_tag", "<Action>"))

        # Load stats saved in checkpoint (or fallback to dataset root in train_config.json)
        stats_candidates = [
            ckpt_path / "stats.json",
            ckpt_path.parent / "stats.json",
        ]
        if ds_cfg.get("root"):
            stats_candidates.append(Path(ds_cfg["root"]) / "meta" / "stats.json")
        stats_file = next((p for p in stats_candidates if p.is_file()), None)
        if stats_file is None:
            raise FileNotFoundError(f"Could not locate stats.json near {ckpt_path}")

        raw_stats = load_json(stats_file)
        if "galbot_one_golf" in raw_stats and isinstance(raw_stats["galbot_one_golf"], dict):
            raw_stats = raw_stats["galbot_one_golf"]
        stats = cast_stats_to_numpy(raw_stats)

        schema = get_schema("galbot_one_golf")
        action_mask = schema.action_mask.numpy().astype(bool)
        action_dim = int(np.asarray(stats[ACTION]["mean"]).reshape(-1).shape[0])

        state_normalizer = NormalizeTransformFn(
            selected_keys=[OBS_STATE],
            mode=norm_mode,
            norm_stats={OBS_STATE: stats[OBS_STATE]},
        )
        processor = InternVLAA15ChatProcessorTransformFn(
            pretrained_model_name_or_path=vlm_model_path or cfg.vlm_model_name_or_path,
            max_length=max_prompt_length,
            num_views=4 if enable_head_history else 3,
            tokenize_state=cfg.tokenize_state,
            max_state_dim=cfg.max_state_dim,
            norm_mode=norm_mode,
            vlash_condition_on_delay=vlash_condition_on_delay,
            eval_output_tag=eval_output_tag,
            use_fast_action_tokens=False,
            mode="eval",
            action_mode=getattr(schema, "action_mode", "joint"),
        )
        compute_dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32

        return cls(
            policy=policy,
            processor=processor,
            state_normalizer=state_normalizer,
            action_stats=stats[ACTION],
            norm_mode=norm_mode,
            action_mode=action_mode,
            action_mask=action_mask,
            chunk_size=int(cfg.chunk_size),
            action_dim=action_dim,
            device=device,
            dtype=compute_dtype,
            fps=fps,
            max_delay_steps=vlash_max_delay,
            fixed_delay_steps=fixed_delay_steps,
            servo_lag_steps=servo_lag_steps,
            action_quant_ratio=action_quant_ratio,
            enable_head_history=enable_head_history,
            head_history_gap=head_history_gap,
            sdk_right_arm_first=sdk_right_arm_first,
            default_task=default_task,
        )

    def reset(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        with self._lock:
            self.controller.reset()
            self._head_history.clear()
            self._last_error = None

    @staticmethod
    def _to_chw_float_tensor(img: np.ndarray | torch.Tensor, target_hw: int = 224) -> torch.Tensor:
        from lerobot.transforms.utils import resize_with_pad

        if isinstance(img, np.ndarray):
            t = torch.from_numpy(img)
            if t.ndim == 3 and t.shape[-1] == 3:
                t = t.permute(2, 0, 1)
            t = t.float()
            if float(t.max()) > 1.5:
                t = t / 255.0
        else:
            t = img.detach().cpu().float()
            if t.ndim == 3 and t.shape[-1] == 3:
                t = t.permute(2, 0, 1)
            if float(t.max()) > 1.5:
                t = t / 255.0
        return resize_with_pad(t, target_hw, target_hw)

    def _run_inference_sync(
        self,
        images_rgb: dict[str, np.ndarray | torch.Tensor],
        future_state: np.ndarray,
        delay_steps: int,
        task: str | None = None,
    ) -> tuple[np.ndarray, float]:
        from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

        head = self._to_chw_float_tensor(images_rgb["head_right"])
        left = self._to_chw_float_tensor(images_rgb["left_arm"])
        right = self._to_chw_float_tensor(images_rgb["right_arm"])

        sample: dict[str, Any] = {
            f"{OBS_IMAGES}.image0": head,
            f"{OBS_IMAGES}.image0_mask": True,
            f"{OBS_IMAGES}.image1": left,
            f"{OBS_IMAGES}.image1_mask": True,
            f"{OBS_IMAGES}.image2": right,
            f"{OBS_IMAGES}.image2_mask": True,
            OBS_STATE: torch.from_numpy(np.asarray(future_state, dtype=np.float32)),
            "task": task or self.default_task,
            "vlash_delay_steps": torch.tensor(delay_steps, dtype=torch.long),
        }
        if self.enable_head_history:
            prev_head = self._head_history[0] if len(self._head_history) > 0 else head
            sample[f"{OBS_IMAGES}.image3"] = prev_head
            sample[f"{OBS_IMAGES}.image3_mask"] = True

        sample = self.state_normalizer(sample)
        sample = self.processor(sample)

        inputs: dict[str, Any] = {}
        for k, v in sample.items():
            if not isinstance(v, torch.Tensor):
                continue
            if k in ("observation.pixel_values", "observation.image_grid_thw"):
                inputs[k] = v.to(self.device)
            elif v.dtype in (torch.int64, torch.int32, torch.bool):
                inputs[k] = v.unsqueeze(0).to(self.device)
            else:
                inputs[k] = v.unsqueeze(0).to(device=self.device, dtype=self.dtype)

        t0 = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        with torch.no_grad(), torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=(self.dtype != torch.float32),
        ):
            norm_chunk = self.policy.predict_action_chunk(inputs)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        infer_s = time.perf_counter() - t0

        norm_np = norm_chunk[0, :, : self.action_dim].detach().float().cpu().numpy()
        unnorm_np = unnormalize_vector(norm_np, self.action_stats, mode=self.norm_mode)
        abs_chunk = reconstruct_absolute_actions(
            unnorm_np,
            ref_state=future_state,
            action_mode=self.action_mode,
            action_mask=self.action_mask,
        )
        return abs_chunk, infer_s

    def _background_worker(
        self,
        images_rgb: dict[str, np.ndarray | torch.Tensor],
        plan: VLASHRollForwardResult,
        task: str | None,
    ) -> None:
        try:
            abs_chunk, infer_s = self._run_inference_sync(
                images_rgb=images_rgb,
                future_state=plan.future_state,
                delay_steps=plan.delay_steps,
                task=task,
            )
            with self._lock:
                self.controller.update_latency(infer_s)
                self.controller.commit_predicted_chunk(abs_chunk, plan)
        except Exception as exc:  # pragma: no cover
            logging.exception("[vlash_async_worker_error] %s", exc)
            with self._lock:
                self._last_error = exc
                self.controller._pending_plan = None

    def step_action(
        self,
        images_rgb: dict[str, np.ndarray | torch.Tensor],
        sensor_state: np.ndarray,
        task: str | None = None,
    ) -> np.ndarray:
        """Non-blocking real-time control tick returning a 16-dim absolute joint + gripper target.

        Blocks only on cold start (when the action queue is completely empty); thereafter runs
        inference asynchronously in a background thread using VLASH future-state roll-forward.
        """
        if self._last_error is not None:
            raise RuntimeError("Background VLASH inference failed") from self._last_error

        state_model = (
            swap_dual_arm_order(sensor_state)
            if self.sdk_right_arm_first
            else np.asarray(sensor_state, dtype=np.float32)
        )
        if self.enable_head_history:
            self._head_history.append(self._to_chw_float_tensor(images_rgb["head_right"]))

        with self._lock:
            action = self.controller.pop_action()
            need_trigger = self.controller.should_trigger_inference()
            if action is None:
                # Cold start: synchronous first inference with delay=0
                plan = self.controller.plan_future_state(state_model)
            elif need_trigger:
                plan = self.controller.plan_future_state(state_model)
            else:
                plan = None

        if action is None:
            abs_chunk, infer_s = self._run_inference_sync(
                images_rgb=images_rgb,
                future_state=plan.future_state,
                delay_steps=plan.delay_steps,
                task=task,
            )
            with self._lock:
                self.controller.update_latency(infer_s)
                self.controller.commit_predicted_chunk(abs_chunk, plan)
                action = self.controller.pop_action()
        elif plan is not None:
            self._worker_thread = threading.Thread(
                target=self._background_worker,
                args=(images_rgb, plan, task),
                daemon=True,
            )
            self._worker_thread.start()

        assert action is not None
        return swap_dual_arm_order(action) if self.sdk_right_arm_first else action
