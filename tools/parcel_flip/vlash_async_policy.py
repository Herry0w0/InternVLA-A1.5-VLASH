"""VLASH (Future-State-Aware Asynchronous Inference) Runtime for InternVLA-A1.5.

Implements:
  1. Future-state roll-forward (VLASH, arXiv:2512.01031) aligned with official VLASH action-proxy
     training (`vlash_use_state_ground_truth=False`: `delta == 0` uses sensor `s_t`, `delta > 0`
     uses committed target `a_{t+delta-1}` with zero servo-lag mismatch).
  2. Exact zero-off-by-one control loop timing: plans next async inference at step `t` using `o_t`
     BEFORE popping `a_t`, so `handover_step = t + d` and `future_state = a_{t+d-1}`.
  3. Constant-length eval prompt padding (`eval_pad_length`) + explicit `warmup()` so
     `InternVLAA15Optimized` captures exactly ONE CUDA Graph without live recaptures or VRAM leaks.
  4. Thread-safe background inference with `_infer_lock` (preventing concurrent CUDA Graph buffer
     access when the queue empties) and immutable camera frame snapshots.
  5. Exact `action_quant_ratio` scaling for both `; Delay: <delta>` and trigger thresholds, plus
     optional explicit `n_action_steps` execution horizon (matching official `vlash/run.py`).
"""

from __future__ import annotations

import json
import logging
import math
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


def quantized_chunk_length(chunk_size: int, quant_ratio: int = 1) -> int:
    """Exact number of execution ticks produced by ``quantize_action_chunk``."""
    n = max(1, int(chunk_size))
    q = max(1, int(quant_ratio))
    if q <= 1:
        return n
    return int(math.ceil(n / q))


@dataclass
class VLASHRollForwardResult:
    """Result of planning an asynchronous VLASH inference call.

    Attributes:
        future_state: 16-dim absolute conditioning state (`s_t` on cold start, else `a_{t+d-1}`).
        delay_steps: Delay in model/training step units (`delay_ticks * action_quant_ratio`),
            injected into the prompt as ``; Delay: <delay_steps>``.
        trigger_step: Control-loop tick `t` at which the observation `o_t` was captured.
        handover_step: Control-loop tick `t + delay_ticks` at which the predicted chunk takes over.
        delay_ticks: Delay in control-loop execution ticks (`handover_step - trigger_step`).
    """

    future_state: np.ndarray
    delay_steps: int
    trigger_step: int
    handover_step: int
    delay_ticks: int = 0

    def __post_init__(self) -> None:
        if self.delay_ticks == 0 and self.handover_step > self.trigger_step:
            self.delay_ticks = self.handover_step - self.trigger_step


class VLASHFutureStateController:
    """Pure-Python/NumPy controller for VLASH future-state roll-forward & async chunk handover.

    Maintains:
      - Current global control tick counter ``step``
      - Committed action buffer ``_action_queue`` (each entry is a 16-dim absolute target)
      - Exponential moving average of inference latency ``ema_infer_s``

    Note on ``action_quant_ratio``:
      For dynamic moving-conveyor tasks, keep ``action_quant_ratio = 1`` so the action timeline
      stays 1:1 locked to physical conveyor velocity and ``; Delay: <delta>``. If ``q > 1`` is
      used, both ``delay_steps`` (scaled by ``q``) and trigger thresholds (in quantized ticks)
      are converted consistently following official VLASH commit ``96aafb5``.
    """

    def __init__(
        self,
        *,
        fps: float = 30.0,
        chunk_size: int = 30,
        n_action_steps: int | None = None,
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
        self.n_action_steps = (
            max(1, min(self.chunk_size, int(n_action_steps)))
            if n_action_steps is not None
            else None
        )
        self.min_delay_steps = max(0, int(min_delay_steps))
        self.max_delay_steps = max(0, int(max_delay_steps))
        self.fixed_delay_steps = fixed_delay_steps
        self.servo_lag_steps = max(0, int(servo_lag_steps))
        self.action_quant_ratio = max(1, int(action_quant_ratio))
        self.latency_ema_alpha = float(latency_ema_alpha)
        self.ema_infer_s = float(initial_latency_s)
        # Optional explicit remaining-tick threshold to launch next async inference
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

    @property
    def effective_chunk_ticks(self) -> int:
        return quantized_chunk_length(self.chunk_size, self.action_quant_ratio)

    def estimate_delay_ticks(self) -> int:
        """Estimate inference overlap delay in control-loop execution ticks."""
        q = self.action_quant_ratio
        max_ticks = self.max_delay_steps // q if q > 1 else self.max_delay_steps
        min_ticks = int(math.ceil(self.min_delay_steps / q)) if self.min_delay_steps > 0 else 0
        min_ticks = min(min_ticks, max_ticks)

        if self.fixed_delay_steps is not None:
            raw_ticks = int(self.fixed_delay_steps)
            return max(0, min(max_ticks, raw_ticks))

        est_ticks = int(round(self.ema_infer_s * self.fps))
        return max(min_ticks, min(max_ticks, est_ticks))

    def estimate_delay_steps(self) -> int:
        """Estimate inference delay in model/training step units (ticks * action_quant_ratio)."""
        return min(self.max_delay_steps, self.estimate_delay_ticks() * self.action_quant_ratio)

    def plan_future_state(self, current_sensor_state: np.ndarray) -> VLASHRollForwardResult:
        """Roll forward ``current_sensor_state`` by ``delay_ticks`` committed actions.

        Must be called at tick ``t`` BEFORE ``pop_action()`` pops ``a_t`` so that:
          - ``queue[0]`` is ``a_t``, ..., ``queue[d-1]`` is ``a_{t+d-1}``
          - ``future_state`` is ``a_{t+d-1}`` (when ``servo_lag_steps=0``, matching official VLASH)
          - ``handover_step`` is ``t + d``
          - ``delay_steps`` is ``d * action_quant_ratio`` (in training step units).

        When the committed queue is empty (cold start at step 0), returns ``delay_steps=0``
        and ``future_state=current_sensor_state`` for synchronous first-chunk bootstrap.
        """
        sensor = np.asarray(current_sensor_state, dtype=np.float32).copy()
        desired_ticks = self.estimate_delay_ticks()
        avail = len(self._action_queue)

        if avail == 0 or desired_ticks == 0:
            plan = VLASHRollForwardResult(
                future_state=sensor,
                delay_steps=0,
                trigger_step=self.step,
                handover_step=self.step,
                delay_ticks=0,
            )
            self._pending_plan = plan
            return plan

        effective_ticks = min(desired_ticks, avail)
        model_delay_steps = min(self.max_delay_steps, effective_ticks * self.action_quant_ratio)

        # Index in committed queue whose execution brings robot to tick t + effective_ticks.
        # With official VLASH action-proxy training (vlash_use_state_ground_truth=False),
        # servo_lag_steps=0 is exact (queue[effective_ticks - 1] == a_{t+d-1}).
        queue_idx = effective_ticks - 1 - self.servo_lag_steps
        if queue_idx < 0:
            future_state = sensor
        else:
            future_state = np.asarray(self._action_queue[queue_idx], dtype=np.float32).copy()

        plan = VLASHRollForwardResult(
            future_state=future_state,
            delay_steps=model_delay_steps,
            trigger_step=self.step,
            handover_step=self.step + effective_ticks,
            delay_ticks=effective_ticks,
        )
        self._pending_plan = plan
        return plan

    def should_trigger_inference(self) -> bool:
        """Return True when the controller should launch a new background inference at tick ``t``."""
        if self._pending_plan is not None:
            return False
        avail = len(self._action_queue)
        if avail == 0:
            return True

        est_ticks = self.estimate_delay_ticks()
        chunk_ticks = self.effective_chunk_ticks

        if self.min_remaining_to_trigger is not None:
            threshold = int(self.min_remaining_to_trigger)
        elif self.n_action_steps is not None:
            # Official VLASH semantics: execute n_action_ticks per chunk, launching inference
            # est_ticks before the chunk boundary (at tick n_action_ticks - est_ticks).
            n_ticks = quantized_chunk_length(self.n_action_steps, self.action_quant_ratio)
            n_ticks = max(est_ticks + 1, min(chunk_ticks, n_ticks))
            executed_before_trigger = max(1, n_ticks - est_ticks)
            threshold = max(est_ticks, chunk_ticks - executed_before_trigger)
        else:
            # Leave enough committed ticks to cover inference + 2-tick jitter margin, while
            # strictly capping below chunk_ticks so q > 1 never re-triggers on every tick.
            threshold = min(max(1, chunk_ticks - 1), max(est_ticks + 2, chunk_ticks // 2))

        return avail <= threshold

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
        n_action_steps: int | None = None,
        min_delay_steps: int = 1,
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

        if int(action_quant_ratio) > 1:
            logging.warning(
                "action_quant_ratio=%d > 1 scales action timeline faster than physical conveyor speed; "
                "for dynamic conveyor tasks, action_quant_ratio=1 is strongly recommended.",
                int(action_quant_ratio),
            )

        self.controller = VLASHFutureStateController(
            fps=fps,
            chunk_size=chunk_size,
            n_action_steps=n_action_steps,
            min_delay_steps=min_delay_steps,
            max_delay_steps=max_delay_steps,
            fixed_delay_steps=fixed_delay_steps,
            servo_lag_steps=servo_lag_steps,
            action_quant_ratio=action_quant_ratio,
        )
        self._head_history: deque[torch.Tensor] = deque(maxlen=max(1, head_history_gap + 1))
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None
        self._last_error: Exception | None = None
        self._warmed_up: bool = False

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir: str | Path,
        *,
        vlm_model_path: str | None = None,
        device: str = "cuda",
        inference_backend: str = "optimized",
        w8a16_action_expert: bool = True,
        use_cuda_graph: bool = True,
        fps: float = 30.0,
        n_action_steps: int | None = None,
        fixed_delay_steps: int | None = None,
        servo_lag_steps: int = 0,
        action_quant_ratio: int = 1,
        eval_pad_length: int | None = None,
        auto_warmup: bool = False,
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
        if inference_backend == "flashrt":
            cfg.flashrt_w8a16_action_expert = bool(w8a16_action_expert)
            cfg.flashrt_use_cuda_graph = bool(use_cuda_graph)
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
        vlash_min_delay = int(ds_cfg.get("vlash_min_delay_steps", 1))
        vlash_max_delay = int(ds_cfg.get("vlash_max_delay_steps", 12))
        enable_head_history = bool(ds_cfg.get("enable_head_history", False))
        head_history_gap = int(ds_cfg.get("head_history_gap", 4))
        max_prompt_length = int(ds_cfg.get("max_prompt_length", 650))
        eval_output_tag = str(ds_cfg.get("eval_output_tag", "<Action>"))

        # Constant eval padding length prevents CUDA-Graphed backends from recapturing a new
        # CUDA Graph every time digitized state/delay text token counts fluctuate.
        if eval_pad_length is None and inference_backend in ("optimized", "flashrt"):
            eval_pad_length = 512 if enable_head_history else 416

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
            eval_pad_length=eval_pad_length,
            use_fast_action_tokens=False,
            mode="eval",
            action_mode=getattr(schema, "action_mode", "joint"),
        )
        compute_dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32

        instance = cls(
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
            n_action_steps=n_action_steps,
            min_delay_steps=vlash_min_delay,
            max_delay_steps=vlash_max_delay,
            fixed_delay_steps=fixed_delay_steps,
            servo_lag_steps=servo_lag_steps,
            action_quant_ratio=action_quant_ratio,
            enable_head_history=enable_head_history,
            head_history_gap=head_history_gap,
            sdk_right_arm_first=sdk_right_arm_first,
            default_task=default_task,
        )
        if auto_warmup:
            instance.warmup()
        return instance

    def warmup(self, num_iters: int = 3, task: str | None = None) -> float:
        """Warm up VLM + CUDA Graph replay and initialize ``ema_infer_s`` with steady-state latency."""
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        dummy_images = {
            "head_right": dummy_img,
            "left_arm": dummy_img,
            "right_arm": dummy_img,
        }
        dummy_state = np.zeros(self.action_dim, dtype=np.float32)
        latencies: list[float] = []
        n = max(1, int(num_iters))
        for _ in range(n):
            _, infer_s = self._run_inference_sync(
                images_rgb=dummy_images,
                future_state=dummy_state,
                delay_steps=0,
                task=task or self.default_task,
            )
            latencies.append(infer_s)

        # Exclude first graph-capture iteration when multiple warmup iterations are run
        steady = latencies[1:] if len(latencies) > 1 else latencies
        steady_median = float(np.median(steady))
        with self._lock:
            self.controller.ema_infer_s = steady_median
            self._warmed_up = True
        logging.info(
            "VLASHAsyncInternVLAPolicy warmup complete (%d iters): first=%.1fms steady_median=%.1fms",
            n,
            latencies[0] * 1000.0,
            steady_median * 1000.0,
        )
        return steady_median

    def reset(self) -> None:
        worker = self._worker_thread
        if worker is not None and worker.is_alive():
            worker.join(timeout=5.0)
        with self._lock:
            self.controller.reset()
            self._head_history.clear()
            self._last_error = None
            self._worker_thread = None

    @staticmethod
    def _snapshot_images(
        images_rgb: dict[str, np.ndarray | torch.Tensor],
    ) -> dict[str, np.ndarray | torch.Tensor]:
        """Deep-copy camera frame buffers so SDK reuse cannot tear frames read by worker thread."""
        snap: dict[str, np.ndarray | torch.Tensor] = {}
        for k, v in images_rgb.items():
            if isinstance(v, np.ndarray):
                snap[k] = np.array(v, copy=True)
            elif isinstance(v, torch.Tensor):
                snap[k] = v.detach().clone()
            else:
                snap[k] = v
        return snap

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
        prev_head_frame: torch.Tensor | None = None,
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
            if prev_head_frame is not None:
                prev_head = prev_head_frame
            else:
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

        with self._infer_lock:
            t0 = time.perf_counter()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            # Do NOT wrap InternVLAA15Optimized in torch.amp.autocast: Qwen3.5 VLM already holds
            # native bf16 weights while the CUDA-Graphed Action Expert runs in native fp32, and
            # autocast's ephemeral weight cache leaves dangling pointers across CUDA Graph replays.
            with torch.no_grad():
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
        prev_head_frame: torch.Tensor | None = None,
    ) -> None:
        try:
            abs_chunk, infer_s = self._run_inference_sync(
                images_rgb=images_rgb,
                future_state=plan.future_state,
                delay_steps=plan.delay_steps,
                task=task,
                prev_head_frame=prev_head_frame,
            )
            with self._lock:
                self.controller.update_latency(infer_s)
                self.controller.commit_predicted_chunk(abs_chunk, plan)
                self._warmed_up = True
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

        Blocks only on cold start (or if an in-flight background worker overran the entire queue,
        in which case it safely joins the worker instead of racing on CUDA Graph buffers).

        Timing invariant (zero off-by-one):
          At tick ``t``, ``o_t`` triggers planning BEFORE ``a_t`` is popped, so
          ``future_state = a_{t+d-1}`` and ``handover_step = t + d``, matching training
          ``(o_t, a_{t+d-1}, Delay=d) -> A_{t+d:}``.
        """
        if self._last_error is not None:
            raise RuntimeError("Background VLASH inference failed") from self._last_error

        state_model = (
            swap_dual_arm_order(sensor_state)
            if self.sdk_right_arm_first
            else np.asarray(sensor_state, dtype=np.float32)
        )
        prev_head_snapshot: torch.Tensor | None = None
        if self.enable_head_history:
            self._head_history.append(self._to_chw_float_tensor(images_rgb["head_right"]))
            prev_head_snapshot = self._head_history[0].clone()

        # Check if queue is empty while a background worker is still in flight; if so, join it
        # cleanly rather than racing on InternVLAA15Optimized CUDA Graph static buffers.
        worker_to_join: threading.Thread | None = None
        with self._lock:
            self.controller._maybe_handover()
            if self.controller.queue_length == 0:
                if self._worker_thread is not None and self._worker_thread.is_alive():
                    worker_to_join = self._worker_thread

        if worker_to_join is not None:
            worker_to_join.join()
            if self._last_error is not None:
                raise RuntimeError("Background VLASH inference failed") from self._last_error

        plan_async: VLASHRollForwardResult | None = None
        plan_sync: VLASHRollForwardResult | None = None
        action: np.ndarray | None = None

        with self._lock:
            self.controller._maybe_handover()
            if self.controller.queue_length == 0:
                # Cold start (or fallback): synchronous inference with delay=0
                plan_sync = self.controller.plan_future_state(state_model)
            else:
                # Normal async tick t: plan BEFORE popping a_t so handover_step = t + d
                if self.controller.should_trigger_inference():
                    plan_async = self.controller.plan_future_state(state_model)
                action = self.controller.pop_action()

        if plan_sync is not None:
            abs_chunk, infer_s = self._run_inference_sync(
                images_rgb=images_rgb,
                future_state=plan_sync.future_state,
                delay_steps=plan_sync.delay_steps,
                task=task,
                prev_head_frame=prev_head_snapshot,
            )
            with self._lock:
                # Do not pollute EMA with un-warmed first CUDA Graph capture latency
                if self._warmed_up:
                    self.controller.update_latency(infer_s)
                else:
                    self._warmed_up = True
                self.controller.commit_predicted_chunk(abs_chunk, plan_sync)
                action = self.controller.pop_action()
        elif plan_async is not None:
            images_snapshot = self._snapshot_images(images_rgb)
            worker = threading.Thread(
                target=self._background_worker,
                args=(images_snapshot, plan_async, task, prev_head_snapshot),
                daemon=True,
            )
            with self._lock:
                self._worker_thread = worker
            worker.start()

        assert action is not None
        return swap_dual_arm_order(action) if self.sdk_right_arm_first else action
