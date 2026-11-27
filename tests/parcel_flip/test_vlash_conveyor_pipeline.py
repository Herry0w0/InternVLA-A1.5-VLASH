"""Unit & integration tests for InternVLA-A1.5 + VLASH parcel-flip pipeline (3-cam)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

try:
    import pytest
except ImportError:  # pragma: no cover - allow running standalone without pytest
    pytest = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET_ROOT = Path("/mnt/data3/jiangxirui/dynamic_rigid_parcel_flip_internvla_lerobot_v3_rel30")
YAML_PATH = REPO_ROOT / "configs/parcel_flip/internvla_a15_vlash_conveyor_delta_3cam.yaml"
LAUNCH_SCRIPT = REPO_ROOT / "launch/parcel_flip/train_internvla_a15_vlash_conveyor.sh"


def test_yaml_and_launch_security_and_defaults():
    with YAML_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds = cfg["dataset"]
    pol = cfg["policy"]
    assert ds["enable_head_history"] is False, "Must keep 3-camera input for Orin latency"
    assert ds["norm_mode"] == "mean_std", "Must match InternVLA-A1.5 pretraining normalization"
    assert ds["vlash_enabled"] is True
    assert ds["vlash_condition_on_delay"] is True
    assert ds["vlash_use_state_ground_truth"] is False
    assert ds["wan_video_span_steps"] == 42
    assert ds["letterbox_pad_px"] == 28
    assert ds["eval_output_tag"] == "<Action>"
    assert pol["mask_action_pad"] is True
    assert pol["padded_action_dim_weight"] == 0.1
    assert pol["use_sdpa"] is False
    assert pol["gradient_checkpointing"] is True

    script_text = LAUNCH_SCRIPT.read_text(encoding="utf-8")
    assert "AYD2zteWSNmfQwa12UgFM" not in script_text, "Hardcoded SwanLab API key must be removed"


def test_convert_mcap_left_first_and_uniform_grid():
    # Inspect unified converter module, constants, and uniform grid builder
    import importlib.util

    assert not (REPO_ROOT / "tools/parcel_flip/standalone_convert_mcap_remote.py").exists()
    assert not (REPO_ROOT / "tools/parcel_flip/convert_mcap_to_internvla_v30.py").exists()
    conv_path = REPO_ROOT / "tools/parcel_flip/convert_mcap_to_internvla_lerobot_v3.py"
    src = conv_path.read_text(encoding="utf-8")
    assert 'default="unified"' in src

    async_src = (REPO_ROOT / "tools/parcel_flip/vlash_async_policy.py").read_text(encoding="utf-8")
    assert 'default_task: str = "Flip the parcel so the label faces up"' in async_src

    spec = importlib.util.spec_from_file_location("convert_mcap_module", conv_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["convert_mcap_module"] = mod
    spec.loader.exec_module(mod)
    assert mod.JOINT_GROUPS == ["left_arm", "left_gripper", "right_arm", "right_gripper"]
    assert mod.ARM_STATE_ACTION_NAMES[0] == "left_arm_joint1"
    assert mod.ARM_STATE_ACTION_NAMES[7] == "left_arm_gripper"
    assert mod.ARM_STATE_ACTION_NAMES[8] == "right_arm_joint1"
    assert mod.ARM_STATE_ACTION_NAMES[15] == "right_arm_gripper"
    assert mod.DEFAULT_UNIFIED_TASK == "Flip the parcel so the label faces up"
    assert mod.chunk_and_file_index(1005) == (1, 5)
    assert mod.ACTION_MASK.tolist() == [True] * 7 + [False] + [True] * 7 + [False]

    # Test pure-numpy uniform grid formula matching build_uniform_timestamp_grid
    t0 = 1_000_000_000
    cam_a = np.arange(t0, t0 + 2_000_000_000, 33_333_333, dtype=np.int64)  # ~30Hz
    cam_b = np.arange(t0 + 100_000_000, t0 + 1_900_000_000, 50_000_000, dtype=np.int64)  # 20Hz
    sensor = np.arange(t0 - 50_000_000, t0 + 2_100_000_000, 10_000_000, dtype=np.int64)  # 100Hz
    grid = mod.build_uniform_timestamp_grid([cam_a, cam_b, sensor], fps=30)
    assert grid[0] == max(int(cam_a[0]), int(cam_b[0]), int(sensor[0]))
    assert grid[-1] <= min(int(cam_a[-1]), int(cam_b[-1]), int(sensor[-1]))
    assert np.all(np.diff(grid) == int(round(1_000_000_000.0 / 30.0)))
    assert len(grid) == 53  # 1.75s overlap at 30Hz -> indices 0..52 inclusive


def test_detect_letterbox_borders_multi_aspect_ratios():
    # Direct function test (importable without heavy lerobot dataset init)
    vlash_path = REPO_ROOT / "src/lerobot/datasets/vlash_dataset.py"
    src = vlash_path.read_text(encoding="utf-8")
    ns: dict = {"torch": torch}
    fn_code = src.split("def detect_letterbox_borders(")[1].split("class VLASHLeRobotDataset")[0]
    exec("def detect_letterbox_borders(" + fn_code, ns)
    detect_fn = ns["detect_letterbox_borders"]

    # 4:3 -> 28px top/bottom, including H.264 YUV420p ringing noise up to 0.03
    img_43 = torch.zeros(3, 224, 224)
    img_43[:, 28:196, :] = 0.5
    img_43[:, 26:28, 10:20] = 0.03
    img_43[:, 196:198, 10:20] = 0.03
    assert detect_fn(img_43) == (28, 28, 0, 0)

    # 16:9 -> 49px top/bottom
    img_169 = torch.zeros(3, 224, 224)
    img_169[:, 49:175, :] = 0.7
    assert detect_fn(img_169) == (49, 49, 0, 0)

    # 1:1 -> 0px borders
    img_sq = torch.full((3, 224, 224), 0.3)
    assert detect_fn(img_sq) == (0, 0, 0, 0)

    # Completely dark frame -> guard returns (0, 0, 0, 0)
    assert detect_fn(torch.zeros(3, 224, 224)) == (0, 0, 0, 0)


def test_wan_sdpa_fallback_fp32_dtype_and_window_mask():
    import importlib.util

    attn_path = REPO_ROOT / "src/lerobot/policies/internvla_a1_5/wan/modules/attention.py"
    spec = importlib.util.spec_from_file_location("wan_attention_test", attn_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Verify the fixed fallback handles fp32 q, k, v without dtype mismatch when FA2/FA3 unavailable
    orig_fa2, orig_fa3 = mod.FLASH_ATTN_2_AVAILABLE, mod.FLASH_ATTN_3_AVAILABLE
    try:
        mod.FLASH_ATTN_2_AVAILABLE = False
        mod.FLASH_ATTN_3_AVAILABLE = False
        if torch.cuda.is_available():
            dev = "cuda"
            q = torch.randn(2, 8, 4, 32, device=dev, dtype=torch.float32)
            k = torch.randn(2, 8, 4, 32, device=dev, dtype=torch.float32)
            v = torch.randn(2, 8, 4, 32, device=dev, dtype=torch.float32)
            out = mod.flash_attention(q, k, v, window_size=(2, 2), dtype=torch.bfloat16)
            assert out.shape == q.shape
            assert out.dtype == torch.float32
            assert torch.isfinite(out).all()
        else:
            # Verify source order on CPU-only host
            src = attn_path.read_text(encoding="utf-8")
            block = src.split("if not FLASH_ATTN_3_AVAILABLE and not FLASH_ATTN_2_AVAILABLE:")[1][:200]
            assert "v_t = half(v)" in block
            assert "q_t = half(q).to(v_t.dtype)" in block
            assert "k_t = half(k).to(v_t.dtype)" in block
    finally:
        mod.FLASH_ATTN_2_AVAILABLE = orig_fa2
        mod.FLASH_ATTN_3_AVAILABLE = orig_fa3


def test_vlash_async_controller_roll_forward_and_handover():
    from tools.parcel_flip.vlash_async_policy import (
        VLASHFutureStateController,
        quantize_action_chunk,
        reconstruct_absolute_actions,
        swap_dual_arm_order,
        unnormalize_vector,
    )

    # 1. Dual-arm swap is an involution
    vec = np.arange(16, dtype=np.float32)
    swapped = swap_dual_arm_order(vec)
    assert np.allclose(swapped[:8], vec[8:16])
    assert np.allclose(swapped[8:16], vec[:8])
    assert np.allclose(swap_dual_arm_order(swapped), vec)

    # 2. Unnormalize + delta reconstruction keeps grippers [7, 15] absolute
    stats = {
        "mean": np.zeros(16, dtype=np.float32),
        "std": np.ones(16, dtype=np.float32),
    }
    raw_delta = np.ones((4, 16), dtype=np.float32) * 0.5
    raw_delta[:, 7] = 80.0
    raw_delta[:, 15] = 20.0
    unnorm = unnormalize_vector(raw_delta, stats, mode="mean_std")
    ref = np.full(16, 2.0, dtype=np.float32)
    abs_chunk = reconstruct_absolute_actions(unnorm, ref, action_mode="delta")
    assert np.allclose(abs_chunk[:, :7], 2.5, atol=1e-4)
    assert np.allclose(abs_chunk[:, 7], 80.0, atol=1e-3)
    assert np.allclose(abs_chunk[:, 8:15], 2.5, atol=1e-4)
    assert np.allclose(abs_chunk[:, 15], 20.0, atol=1e-3)

    # 3. Action quantization stride=2
    seq = np.arange(10 * 16, dtype=np.float32).reshape(10, 16)
    q_seq = quantize_action_chunk(seq, quant_ratio=2)
    assert q_seq.shape == (5, 16)
    assert np.allclose(q_seq[0], seq[1])
    assert np.allclose(q_seq[-1], seq[9])

    # 4. VLASH roll-forward + servo lag compensation + seamless handover
    ctrl = VLASHFutureStateController(
        fps=30.0,
        chunk_size=10,
        fixed_delay_steps=4,
        servo_lag_steps=1,
    )
    sensor0 = np.zeros(16, dtype=np.float32)
    # Cold start -> delay = 0
    plan0 = ctrl.plan_future_state(sensor0)
    assert plan0.delay_steps == 0
    chunk0 = np.stack([np.full(16, float(i + 1), dtype=np.float32) for i in range(10)], axis=0)
    ctrl.commit_predicted_chunk(chunk0, plan0)

    # Pop 2 actions -> remaining queue has values [3, 4, 5, 6, 7, 8, 9, 10], step == 2
    a0 = ctrl.pop_action()
    a1 = ctrl.pop_action()
    assert a0 is not None and a1 is not None
    assert float(a0[0]) == 1.0 and float(a1[0]) == 2.0
    assert ctrl.step == 2

    # Plan future state at step 2 with delay=4, servo_lag=1 -> queue_idx = 4 - 1 - 1 = 2 -> value 5.0
    plan1 = ctrl.plan_future_state(sensor0)
    assert plan1.delay_steps == 4
    assert plan1.handover_step == 6
    assert float(plan1.future_state[0]) == 5.0

    # New chunk arrives while robot is still at step 3 (before handover step 6)
    a2 = ctrl.pop_action()
    assert a2 is not None and float(a2[0]) == 3.0
    chunk1 = np.stack([np.full(16, 100.0 + i, dtype=np.float32) for i in range(10)], axis=0)
    ctrl.commit_predicted_chunk(chunk1, plan1)

    # Steps 3, 4, 5 must still come from chunk0 (values 4, 5, 6); step 6 switches to chunk1 (100.0)
    vals = [float(ctrl.pop_action()[0]) for _ in range(4)]
    assert vals == [4.0, 5.0, 6.0, 100.0]


    # 5. Action quantization scaling (q=2) and explicit n_action_steps
    ctrl_q = VLASHFutureStateController(
        fps=30.0,
        chunk_size=30,
        fixed_delay_steps=3,
        action_quant_ratio=2,
    )
    assert ctrl_q.effective_chunk_ticks == 15
    plan_q0 = ctrl_q.plan_future_state(sensor0)
    chunk_q0 = np.stack([np.full(16, float(i), dtype=np.float32) for i in range(30)], axis=0)
    ctrl_q.commit_predicted_chunk(chunk_q0, plan_q0)
    assert ctrl_q.queue_length == 15
    # Must NOT trigger immediately when full quantized chunk (15 ticks) is loaded
    assert ctrl_q.should_trigger_inference() is False
    # Plan at tick 0 with delay_ticks=3, q=2 -> model delay_steps must be 3 * 2 = 6
    plan_q1 = ctrl_q.plan_future_state(sensor0)
    assert plan_q1.delay_ticks == 3
    assert plan_q1.delay_steps == 6
    assert plan_q1.handover_step == 3

    # Explicit n_action_steps=16, fixed_delay_steps=4, q=1 -> triggers when remaining == 30 - (16 - 4) = 18
    ctrl_h = VLASHFutureStateController(
        fps=30.0,
        chunk_size=30,
        n_action_steps=16,
        fixed_delay_steps=4,
        action_quant_ratio=1,
    )
    p0 = ctrl_h.plan_future_state(sensor0)
    ctrl_h.commit_predicted_chunk(chunk_q0, p0)
    for step_idx in range(12):
        assert ctrl_h.should_trigger_inference() is False, f"Should not trigger at step {step_idx}"
        ctrl_h.pop_action()
    # At step 12 (remaining=18), should trigger so handover lands cleanly at step 12 + 4 = 16
    assert ctrl_h.should_trigger_inference() is True
    p1 = ctrl_h.plan_future_state(sensor0)
    assert p1.trigger_step == 12
    assert p1.handover_step == 16
    assert float(p1.future_state[0]) == 15.0  # a_{12 + 4 - 1} = a_{15}


def test_vlash_async_zero_off_by_one_snapshot_and_concurrency():
    import threading
    import time
    from tools.parcel_flip.vlash_async_policy import VLASHAsyncInternVLAPolicy

    calls: list[dict] = []
    active_infer = 0
    max_concurrent = 0
    infer_counter_lock = threading.Lock()
    slow_second_call = threading.Event()

    class DummyPolicy:
        def predict_action_chunk(self, inputs):
            nonlocal active_infer, max_concurrent
            with infer_counter_lock:
                active_infer += 1
                max_concurrent = max(max_concurrent, active_infer)
                call_idx = len(calls) - 1
            try:
                if call_idx in (1, 2):
                    # Simulate slow background inference
                    slow_second_call.wait(timeout=0.5)
                    time.sleep(0.03)
                # Encode call_idx * 100 + k into action values
                vals = torch.arange(10, dtype=torch.float32)[:, None].expand(10, 16) + float(call_idx * 100)
                return vals.unsqueeze(0)
            finally:
                with infer_counter_lock:
                    active_infer -= 1

    class DummyNorm:
        def __call__(self, sample):
            return sample

    class DummyProc:
        def __call__(self, sample):
            calls.append(
                {
                    "delay": int(sample["vlash_delay_steps"].item()),
                    "future_state0": float(sample["observation.state"][0].item()),
                    "head_mean": float(sample["observation.images.image0"].mean().item()),
                }
            )
            return {
                "observation.input_ids": torch.ones(8, dtype=torch.long),
                "observation.attention_mask": torch.ones(8, dtype=torch.bool),
                "observation.state": sample["observation.state"],
            }

    wrapper = VLASHAsyncInternVLAPolicy(
        policy=DummyPolicy(),
        processor=DummyProc(),
        state_normalizer=DummyNorm(),
        action_stats={"mean": np.zeros(16, dtype=np.float32), "std": np.ones(16, dtype=np.float32)},
        norm_mode="mean_std",
        action_mode="abs",
        action_mask=np.array([True] * 7 + [False] + [True] * 7 + [False]),
        chunk_size=10,
        action_dim=16,
        n_action_steps=8,
        min_delay_steps=1,
        max_delay_steps=6,
        fixed_delay_steps=3,
        servo_lag_steps=0,
        action_quant_ratio=1,
        device="cpu",
        dtype=torch.float32,
        fps=30.0,
    )
    init_ema = wrapper.controller.ema_infer_s

    # Tick 0 (cold start): delay=0, returns chunk0[0]=0.0, does NOT pollute ema_infer_s
    img_buf = np.zeros((224, 224, 3), dtype=np.uint8)
    images = {"head_right": img_buf, "left_arm": img_buf, "right_arm": img_buf}
    sensor = np.full(16, -5.0, dtype=np.float32)
    a0 = wrapper.step_action(images, sensor)
    assert float(a0[0]) == 0.0
    assert calls[0]["delay"] == 0
    assert calls[0]["future_state0"] == -5.0
    assert wrapper.controller.ema_infer_s == init_ema  # un-warmed cold start did not corrupt EMA

    # Ticks 1..4: consume chunk0[1..4]; n_action_steps=8, delay=3 -> trigger at tick 8 - 3 = 5!
    for t in range(1, 5):
        at = wrapper.step_action(images, sensor)
        assert np.isclose(float(at[0]), float(t), atol=1e-3)
    assert len(calls) == 1  # Not triggered yet

    # Tick 5: set distinctive image value (0.5 -> ~128), call step_action, then IMMEDIATELY overwrite buffer!
    img_buf[:] = 128
    a5 = wrapper.step_action(images, sensor)
    img_buf[:] = 255  # Mutate caller buffer immediately after return!
    assert np.isclose(float(a5[0]), 5.0, atol=1e-3)  # Still executes chunk0[5]

    # Release slow worker and verify ticks 6, 7 execute chunk0[6..7], tick 8 hands over to chunk1[0]=100.0
    slow_second_call.set()
    if wrapper._worker_thread is not None:
        wrapper._worker_thread.join(timeout=2.0)

    assert len(calls) == 2
    assert calls[1]["delay"] == 3
    # Zero off-by-one: triggered at tick 5 with delay 3 -> future_state must be a_{5+3-1} = a_7 = 7.0!
    assert np.isclose(calls[1]["future_state0"], 7.0, atol=1e-3)
    # Snapshot isolation: worker must see ~128/255 (~0.502), NOT overwritten 255/255 (1.0)!
    assert abs(calls[1]["head_mean"] - (128.0 / 255.0)) < 0.02

    a6 = wrapper.step_action(images, sensor)
    a7 = wrapper.step_action(images, sensor)
    a8 = wrapper.step_action(images, sensor)
    assert np.allclose([float(a6[0]), float(a7[0]), float(a8[0])], [6.0, 7.0, 100.0], atol=1e-3)

    # Overrun concurrency test: keep call 2 blocked past the entire remaining queue so step_action
    # empties the queue while background worker is still alive -> must join cleanly (max_concurrent == 1)
    slow_second_call.clear()
    # Consume ticks 9..17 (remaining 9 actions of chunk1); trigger fires at tick 13, worker blocks until delayed release
    def _delayed_release():
        time.sleep(0.08)
        slow_second_call.set()

    releaser = threading.Thread(target=_delayed_release, daemon=True)
    releaser.start()
    overrun_vals = [float(wrapper.step_action(images, sensor)[0]) for _ in range(10)]
    releaser.join(timeout=1.0)
    assert overrun_vals[-1] >= 199.9  # Switched cleanly to chunk2 after joining slow worker
    assert max_concurrent == 1  # Strictly serialized inference, zero CUDA Graph race


def test_action_loss_masking_logic():
    # Standalone verification of InternVLAA15Policy._reduce_action_loss math
    from lerobot.policies.internvla_a1_5.modeling_internvla_a1_5 import InternVLAA15Policy

    dummy = SimpleNamespace(
        config=SimpleNamespace(mask_action_pad=True, padded_action_dim_weight=0.1)
    )
    losses = torch.ones(2, 4, 32, dtype=torch.float32)
    losses[:, :, 16:] = 10.0  # padded dims have high loss
    losses[:, 3, :] = 100.0  # padded tail timestep has huge loss

    action_is_pad = torch.zeros(2, 4, dtype=torch.bool)
    action_is_pad[:, 3] = True
    action_dim_mask = torch.zeros(2, 32, dtype=torch.bool)
    action_dim_mask[:, :16] = True

    batch = {"action_is_pad": action_is_pad, "action_dim_mask": action_dim_mask}
    loss_fm, loss_real = InternVLAA15Policy._reduce_action_loss(dummy, losses, batch)
    # Real dims on valid timesteps [0..2] all have loss=1.0
    assert torch.isclose(loss_real, torch.tensor(1.0), atol=1e-5)
    # Weighted mean across valid timesteps: (16*1.0*1.0 + 16*0.1*10.0) / (16*1.0 + 16*0.1) = 32 / 17.6
    expected_fm = (16.0 * 1.0 + 16.0 * 1.0) / (16.0 * 1.0 + 16.0 * 0.1)
    assert torch.isclose(loss_fm, torch.tensor(expected_fm), atol=1e-5)


if pytest is not None:

    def test_galbot_schema_registration():
        from lerobot.dataset_schemas import get_schema

        schema = get_schema("galbot_one_golf")
        assert schema.robot_type == "galbot_one_golf"
        expected_mask = [True] * 7 + [False] + [True] * 7 + [False]
        assert schema.action_mask.tolist() == expected_mask
        assert schema.image_mapping["observation.images.head_right"] == "observation.images.image0"

    @pytest.mark.skipif(not (DATASET_ROOT / "meta/stats.json").exists(), reason="Converted dataset not present")
    def test_relative_stats_integrity():
        with (DATASET_ROOT / "meta/stats.json").open("r", encoding="utf-8") as f:
            stats = json.load(f)

        for key in ("observation.state", "action"):
            assert key in stats
            q01 = np.asarray(stats[key]["q01"], dtype=np.float64)
            q99 = np.asarray(stats[key]["q99"], dtype=np.float64)
            std = np.asarray(stats[key]["std"], dtype=np.float64)
            assert q01.shape == (16,)
            assert np.all(np.isfinite(q01)) and np.all(np.isfinite(q99)) and np.all(np.isfinite(std))
            assert np.all((q99 - q01) >= 0.05), f"Degenerate quantile range in {key}: {q99 - q01}"
            assert np.all(std >= 0.01), f"Degenerate std in {key}: {std}"

    @pytest.mark.skipif(not (DATASET_ROOT / "meta/info.json").exists(), reason="Converted dataset not present")
    def test_vlash_dataset_3cam_alignment_fixed_wan_span_and_letterbox():
        from lerobot.datasets.transforms import ImageTransformConfig, ImageTransforms, ImageTransformsConfig
        from lerobot.datasets.vlash_dataset import VLASHLeRobotDataset

        img_tf = ImageTransforms(
            ImageTransformsConfig(
                enable=True,
                max_num_transforms=2,
                random_order=False,
                tfs={
                    "brightness": ImageTransformConfig(weight=1.0, type="ColorJitter", kwargs={"brightness": (0.85, 1.15)}),
                    "hue": ImageTransformConfig(weight=1.0, type="ColorJitter", kwargs={"hue": (-0.03, 0.03)}),
                },
            )
        )
        ds = VLASHLeRobotDataset(
            repo_id="local/dynamic_parcel_flip",
            root=DATASET_ROOT,
            image_transforms=img_tf,
            vlash_enabled=True,
            vlash_min_delay_steps=1,
            vlash_max_delay_steps=12,
            vlash_zero_delay_prob=0.0,
            wan_video_span_steps=42,
            enable_head_history=False,
            chunk_size=30,
            num_video_frames=4,
            letterbox_pad_px=-1,
        )

        sample = ds[10]
        delay = int(sample["vlash_delay_steps"].item())
        assert 1 <= delay <= 12
        assert ds.wan_video_span_steps == 42

        ep0_end = int(ds.meta.episodes[0]["dataset_to_index"])
        t_exec = min(ep0_end - 1, 10 + delay)
        # Official VLASH default (vlash_use_state_ground_truth=False): delta>0 uses a_{t+delta-1}
        expected_state = ds.hf_dataset[t_exec - 1]["action"]
        expected_action0 = ds.hf_dataset[t_exec]["action"]

        assert torch.allclose(sample["observation.state"], expected_state, atol=1e-6)
        assert sample["action"].shape == (30, 16)
        assert sample["action_is_pad"].shape == (30,)
        assert torch.allclose(sample["action"][0], expected_action0, atol=1e-6)
        assert sample["observation.video_frames_raw"].shape == (5, 3, 224, 224)
        assert "observation.images.head_right_prev" not in sample

        # Verify delta == 0 still uses recorded sensor s_t (for synchronous cold start)
        ds_zero = VLASHLeRobotDataset(
            repo_id="local/dynamic_parcel_flip",
            root=DATASET_ROOT,
            vlash_enabled=True,
            vlash_min_delay_steps=1,
            vlash_max_delay_steps=12,
            vlash_zero_delay_prob=1.0,
            vlash_use_state_ground_truth=False,
            wan_video_span_steps=42,
            enable_head_history=False,
            chunk_size=30,
            num_video_frames=4,
            letterbox_pad_px=28,
        )
        sample_zero = ds_zero[10]
        assert int(sample_zero["vlash_delay_steps"].item()) == 0
        assert torch.allclose(
            sample_zero["observation.state"], ds_zero.hf_dataset[10]["observation.state"], atol=1e-6
        )

        # Verify optional vlash_use_state_ground_truth=True fallback still works
        ds_gt = VLASHLeRobotDataset(
            repo_id="local/dynamic_parcel_flip",
            root=DATASET_ROOT,
            vlash_enabled=True,
            vlash_min_delay_steps=1,
            vlash_max_delay_steps=12,
            vlash_zero_delay_prob=0.0,
            vlash_use_state_ground_truth=True,
            wan_video_span_steps=42,
            enable_head_history=False,
            chunk_size=30,
            num_video_frames=4,
            letterbox_pad_px=28,
        )
        sample_gt = ds_gt[10]
        t_exec_gt = min(ep0_end - 1, 10 + int(sample_gt["vlash_delay_steps"].item()))
        assert torch.allclose(
            sample_gt["observation.state"], ds_gt.hf_dataset[t_exec_gt]["observation.state"], atol=1e-6
        )

    @pytest.mark.skipif(not (DATASET_ROOT / "meta/info.json").exists(), reason="Converted dataset not present")
    def test_internvla_transforms_delay_prompt_and_conditional_state_dropout(monkeypatch):
        from lerobot.datasets.transformed_dataset import TransformedLeRobotDataset
        from lerobot.datasets.vlash_dataset import VLASHLeRobotDataset
        from lerobot.policies.internvla_a1_5.configuration_internvla_a1_5 import InternVLAA15DatasetConfig

        vlm_path = Path("/mnt/data3/Qwen3.5-2B")
        fast_path = Path("/mnt/data3/fast_tokenizer")
        if not vlm_path.exists() or not fast_path.exists():
            pytest.skip("Offline VLM or FAST tokenizer weights not present")

        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

        ds_cfg = InternVLAA15DatasetConfig(
            repo_id="local/dynamic_parcel_flip",
            root=str(DATASET_ROOT),
            action_mode="delta",
            norm_mode="mean_std",
            state_dropout_prob=1.0,  # would drop 100% if delay==0, must NEVER drop when delay>0!
            chunk_size=30,
            max_prompt_length=650,
            tokenize_state=True,
            use_fast_action_tokens=True,
            vlm_model_name_or_path=str(vlm_path),
            fast_tokenizer_path=str(fast_path),
            vlash_enabled=True,
            vlash_min_delay_steps=1,
            vlash_max_delay_steps=12,
            vlash_zero_delay_prob=0.0,
            vlash_condition_on_delay=True,
            wan_video_span_steps=42,
            enable_head_history=False,
            letterbox_pad_px=-1,
        )
        base_ds = VLASHLeRobotDataset(
            repo_id="local/dynamic_parcel_flip",
            root=DATASET_ROOT,
            vlash_enabled=True,
            vlash_min_delay_steps=1,
            vlash_max_delay_steps=12,
            vlash_zero_delay_prob=0.0,
            wan_video_span_steps=42,
            enable_head_history=False,
            chunk_size=30,
            num_video_frames=4,
            letterbox_pad_px=-1,
        )
        tds = TransformedLeRobotDataset.from_base(base_ds, ds_cfg.data_transforms.inputs)
        out = tds[10]

        assert out["observation.state"].shape == (32,)
        assert out["action"].shape == (30, 32)
        assert out["action_is_pad"].shape == (30,)
        assert out["action_dim_mask"].shape == (32,)
        assert int(out["action_dim_mask"].sum().item()) == 16
        assert out["observation.video_frames"].shape == (5, 3, 224, 224)
        assert out["observation.image_grid_thw"].shape == (3, 3)

        from transformers.models.qwen3_5 import Qwen3_5Tokenizer

        tok = Qwen3_5Tokenizer.from_pretrained(str(vlm_path), local_files_only=True)
        decoded = tok.decode(out["observation.input_ids"].tolist(), skip_special_tokens=False)
        delay = int(out["vlash_delay_steps"].item())
        assert f"Delay: {delay}" in decoded
        assert "State:" in decoded
        assert " -1 " not in decoded

    def test_eval_pad_length_keeps_constant_prefix_len(monkeypatch):
        from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (
            InternVLAA15ChatProcessorTransformFn,
        )

        vlm_path = Path("/mnt/data3/Qwen3.5-2B")
        if not vlm_path.exists():
            pytest.skip("Offline VLM weights not present")
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

        proc = InternVLAA15ChatProcessorTransformFn(
            pretrained_model_name_or_path=str(vlm_path),
            max_length=650,
            num_views=3,
            tokenize_state=True,
            max_state_dim=32,
            norm_mode="mean_std",
            vlash_condition_on_delay=True,
            eval_output_tag="<Action>",
            eval_pad_length=416,
            use_fast_action_tokens=False,
            mode="eval",
            action_mode="joint",
        )
        seen_lens = set()
        for val, delay in [(-3.0, 0), (-1.2, 3), (0.0, 9), (2.9, 12)]:
            sample = {
                "observation.images.image0": torch.zeros(3, 224, 224),
                "observation.images.image0_mask": True,
                "observation.images.image1": torch.zeros(3, 224, 224),
                "observation.images.image1_mask": True,
                "observation.images.image2": torch.zeros(3, 224, 224),
                "observation.images.image2_mask": True,
                "observation.state": torch.full((16,), val),
                "task": "Flip the parcel so the label faces up",
                "vlash_delay_steps": torch.tensor(delay, dtype=torch.long),
            }
            out = proc(sample)
            seen_lens.add(int(out["observation.input_ids"].shape[0]))
            assert int(out["observation.attention_mask"][-1].item()) == 0
        assert seen_lens == {416}


if __name__ == "__main__":
    test_yaml_and_launch_security_and_defaults()
    test_convert_mcap_left_first_and_uniform_grid()
    test_detect_letterbox_borders_multi_aspect_ratios()
    test_wan_sdpa_fallback_fp32_dtype_and_window_mask()
    test_vlash_async_controller_roll_forward_and_handover()
    test_vlash_async_zero_off_by_one_snapshot_and_concurrency()
    print("Standalone pure-Python unit tests PASSED!")
