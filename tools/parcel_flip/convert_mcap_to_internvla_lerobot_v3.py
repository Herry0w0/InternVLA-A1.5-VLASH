#!/usr/bin/env python3
"""Unified Galbot MCAP -> LeRobot v3.0 + VLASH Relative Stats converter for InternVLA-A1.5 + VLASH.

Standalone & out-of-the-box (no torch, no lerobot, no external ffprobe binary required):
    pip install mcap protobuf av opencv-python numpy pandas pyarrow pyyaml tqdm

What this script does in one unified pipeline (`--step convert,finalize,verify`):
  1. Discovers all *.FIN.mcap / *.SYNC.mcap (preferring .FIN.mcap over .SYNC.mcap for the
     same stem; ignoring macOS '._*' AppleDouble files & dotfiles) under task subdirectories
     such as `D:\\dynamic_parcel_flip_data` (`rigid_label_down`, `rigid_label_left_back`,
     `rigid_label_right_front`, `rigid_label_up`). Supports `--exclude-files` and
     `--max-episodes-per-folder` for smoke testing.
  2. Assigns task prompts using `--task-mode unified` by default (`"Flip the parcel so the
     label faces up"` across all orientation folders, matching `vlash_async_policy.py`), or
     `--task-mode per_folder` with `--task-map-yaml`.
  3. Extracts 16-dim dual-arm state & action in InternVLA-A1.5 pretraining layout
     (left arm first):
       left_arm(7) [0:7] + left_gripper(1) [7:8] + right_arm(7) [8:15] + right_gripper(1) [15:16]
     Strictly validates sensor & target group dimensions (never silently zero-fills missing
     sensor groups).
  4. Resamples 3 cameras (`head_right`, `left_arm`, `right_arm`) and sensor state onto a true
     uniform constant-FPS grid (default 30Hz) over the exact overlapping time window, then
     applies Causal Zero-Order Hold (Causal ZOH) target alignment with sequential-bimanual
     startup grace, hold-consistency rescue, interior gap splitting, and pre-start state hold.
  5. Resizes frames to 224x224 via Letterbox (`--resize-mode pad`) using `cv2.INTER_LINEAR`
     (matching PyTorch `bilinear(align_corners=False)` in `resize_with_pad` at inference),
     atomically encodes H.264 MP4s (`libx264`, `crf=18`, `g=2`, `preset=veryfast`, `yuv420p`)
     via PyAV, and verifies frame count, resolution, PTS error < 1e-4s, first-frame MAE,
     and letterbox border blackness.
  6. Supports `chunks_size = 1000` directory sharding (`chunk-000/file-000`, `chunk-001/...`),
     multiprocessing (`--workers`), atomic `progress.json` with `config_fingerprint` resume
     protection, and exact `float64` VLASH relative-action statistics (`chunk_size=30`,
     arm joints relative to `s_t`, grippers [7, 15] absolute, excluding masked padded tail
     by default to match `mask_action_pad: true`), plus stationary-dimension protection and
     a built-in `verify` stage + `--self-test`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from mcap.reader import SeekingReader

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

# ============================================================================
# 1. Constants & Training Contract
# ============================================================================
CODEBASE_VERSION = "v3.0"
CHUNKS_SIZE = 1000
PROGRESS_SCHEMA_VERSION = 3
SKIP_EMPTY_ACTIONS = True

OBS_STATE = "observation.state"
ACTION = "action"

QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]
QUANTILE_KEYS = [f"q{int(q * 100):02d}" for q in QUANTILES]

CAMERA_TOPICS: dict[str, str] = {
    "observation.images.head_right": "/front_head_camera/right_color/image_raw",
    "observation.images.left_arm": "/left_arm_camera/color/image_raw",
    "observation.images.right_arm": "/right_arm_camera/color/image_raw",
}

# Left arm first — aligned with InternVLA-A1.5 pretraining & galbot_one_golf schema
JOINT_GROUPS: list[str] = ["left_arm", "left_gripper", "right_arm", "right_gripper"]
DIM_MAPS: dict[str, int] = {
    "left_arm": 7,
    "left_gripper": 1,
    "right_arm": 7,
    "right_gripper": 1,
}
ACTION_MASK = np.array([True] * 7 + [False] + [True] * 7 + [False], dtype=bool)
SENSOR_TOPICS = ["singorix/wbcs/sensor"]
ACTION_TOPICS = ["singorix/wbcs/target"]

ARM_STATE_ACTION_NAMES: list[str] = [
    "left_arm_joint1",
    "left_arm_joint2",
    "left_arm_joint3",
    "left_arm_joint4",
    "left_arm_joint5",
    "left_arm_joint6",
    "left_arm_joint7",
    "left_arm_gripper",
    "right_arm_joint1",
    "right_arm_joint2",
    "right_arm_joint3",
    "right_arm_joint4",
    "right_arm_joint5",
    "right_arm_joint6",
    "right_arm_joint7",
    "right_arm_gripper",
]

DEFAULT_UNIFIED_TASK = "Flip the parcel so the label faces up"

DEFAULT_TASK_MAP: dict[str, str] = {
    "rigid_label_down": "Flip the parcel label up",
    "rigid_label_left_back": "Flip the parcel label up",
    "rigid_label_right_front": "Flip the parcel label up",
    "soft_label_down": "Flip the parcel label up",
    "rigid_label_up": "Slide the parcel directly",
}

DEFAULT_FEATURES: dict[str, dict[str, Any]] = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}

DEFAULT_FPS = 30
DEFAULT_TARGET_RESOLUTION = "224x224"
DEFAULT_CODEC = "libx264"
DEFAULT_CRF = 18
DEFAULT_GOP = 2
DEFAULT_PRESET = "veryfast"
DEFAULT_MAX_FIRST_FRAME_MAE = 12.0
DEFAULT_MAX_PAD_MEAN = 8.0
DEFAULT_PTS_TOLERANCE_S = 1e-4

DEFAULT_CHUNK_SIZE = 30
DEFAULT_MIN_FRAMES = 45  # >= wan_video_span_steps (42) and chunk_size (30)
DEFAULT_ARM_TOLERANCE_MS = 200.0
DEFAULT_ARM_STARTUP_GRACE_S = 1.5
DEFAULT_ARM_STARTUP_MAX_GAP_MS = 1000.0
DEFAULT_HOLD_EPS_RAD = 0.05
DEFAULT_ARM_MIN_RANGE = 0.1
DEFAULT_ARM_MIN_STD = 0.02
EXACT_QUANTILE_ROW_LIMIT = 5_000_000


def _default_source_root() -> Path:
    if os.name == "nt":
        return Path(r"D:\dynamic_parcel_flip_data")
    return Path("/mnt/data3/jiangxirui/dynamic_parcel_flip_data")


def _default_output_dir() -> Path:
    if os.name == "nt":
        return Path(r"D:\dynamic_rigid_parcel_flip_internvla_lerobot_v3_rel30")
    return Path("/mnt/data3/jiangxirui/dynamic_rigid_parcel_flip_internvla_lerobot_v3_rel30")


# ============================================================================
# 2. Data Structures & Basic Utilities
# ============================================================================
@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int
    duration: float
    codec: str
    pix_fmt: str
    has_audio: bool


@dataclass(frozen=True)
class VideoVerifyMetrics:
    camera_key: str
    input_width: int
    input_height: int
    output_width: int
    output_height: int
    output_fps: float
    output_frames: int
    first_frame_mae: float
    pad_mean: float
    max_pts_err_s: float


def chunk_and_file_index(ep_index: int, chunks_size: int = CHUNKS_SIZE) -> tuple[int, int]:
    return ep_index // chunks_size, ep_index % chunks_size


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON via a sibling temporary file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)
        f.write("\n")
    os.replace(tmp_path, path)


def parse_target_resolution(value: str) -> tuple[int, int] | None:
    text = str(value or "").strip().lower()
    if text in ("keep", "none", "0x0", ""):
        return None
    match = re.fullmatch(r"(\d+)\s*[x×]\s*(\d+)", text)
    if not match:
        raise ValueError(f"Invalid target resolution: {value!r}")
    return int(match.group(1)), int(match.group(2))


# ============================================================================
# 3. Video Resize (INTER_LINEAR), PyAV Encoding, Stats & Verification
# ============================================================================
def compute_letterbox_padding(src_w: int, src_h: int, target_w: int, target_h: int) -> dict[str, int]:
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    return {
        "src_w": src_w,
        "src_h": src_h,
        "new_w": new_w,
        "new_h": new_h,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
    }


def _padding_slices(src_w: int, src_h: int, target_w: int, target_h: int) -> list[tuple[slice, slice]]:
    if (src_w, src_h) == (target_w, target_h):
        return []
    info = compute_letterbox_padding(src_w, src_h, target_w, target_h)
    regions: list[tuple[slice, slice]] = []
    if info["pad_top"] > 0:
        regions.append((slice(0, info["pad_top"]), slice(None)))
    if info["pad_bottom"] > 0:
        regions.append((slice(target_h - info["pad_bottom"], target_h), slice(None)))
    if info["pad_left"] > 0:
        regions.append((slice(None), slice(0, info["pad_left"])))
    if info["pad_right"] > 0:
        regions.append((slice(None), slice(target_w - info["pad_right"], target_w)))
    return regions


def resize_bgr_frame(img: np.ndarray, target_size: tuple[int, int], resize_mode: str) -> np.ndarray:
    """Resize BGR uint8 frame using INTER_LINEAR (matches PyTorch bilinear align_corners=False)."""
    target_w, target_h = target_size
    src_h, src_w = img.shape[:2]
    if (src_w, src_h) == (target_w, target_h):
        return img
    if resize_mode == "pad":
        info = compute_letterbox_padding(src_w, src_h, target_w, target_h)
        resized = cv2.resize(img, (info["new_w"], info["new_h"]), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x0, y0 = info["pad_left"], info["pad_top"]
        canvas[y0 : y0 + info["new_h"], x0 : x0 + info["new_w"]] = resized
        return canvas
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _stream_fps(stream: av.video.stream.VideoStream) -> Fraction:
    rate = stream.average_rate or stream.guessed_rate or stream.base_rate
    if rate is None or float(rate) <= 0:
        raise ValueError("Video stream has no valid frame rate")
    return Fraction(rate)


def probe_video_pyav(path: Path) -> VideoInfo:
    """Probe MP4 metadata purely with PyAV (zero external ffprobe binary)."""
    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in {path}")
        stream = container.streams.video[0]
        width, height = int(stream.width), int(stream.height)
        fps = float(_stream_fps(stream))
        codec_name = str(stream.codec_context.name or "")
        pix_fmt = str(stream.codec_context.pix_fmt or "")
        has_audio = bool(container.streams.audio)
        frames = int(stream.frames or 0)
        if frames <= 0:
            frames = sum(1 for _ in container.decode(stream))
        duration = float(frames) / fps if fps > 0 else 0.0
    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        frames=frames,
        duration=duration,
        codec=codec_name,
        pix_fmt=pix_fmt,
        has_audio=has_audio,
    )


def _sample_video_indices(n_frames: int) -> list[int]:
    min_num = min(100, n_frames)
    num_samples = max(min_num, min(int(n_frames**0.75), 10_000))
    return np.round(np.linspace(0, n_frames - 1, num_samples)).astype(int).tolist()


def _auto_downsample_chw(img_chw: np.ndarray, target_size: int = 150, max_size_threshold: int = 300) -> np.ndarray:
    _, height, width = img_chw.shape
    if max(width, height) < max_size_threshold:
        return img_chw
    factor = int(width / target_size) if width > height else int(height / target_size)
    factor = max(1, factor)
    return img_chw[:, ::factor, ::factor]


def _compute_image_channel_stats(sampled_chw_uint8: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Compute per-channel stats with shape (3, 1, 1) normalized to [0, 1]."""
    arr = np.stack(sampled_chw_uint8, axis=0).astype(np.float64) / 255.0  # [S, 3, H, W]
    q_vals = np.quantile(arr, QUANTILES, axis=(0, 2, 3), keepdims=True)
    stats: dict[str, np.ndarray] = {
        "min": np.min(arr, axis=(0, 2, 3), keepdims=True).squeeze(0),
        "max": np.max(arr, axis=(0, 2, 3), keepdims=True).squeeze(0),
        "mean": np.mean(arr, axis=(0, 2, 3), keepdims=True).squeeze(0),
        "std": np.std(arr, axis=(0, 2, 3), keepdims=True).squeeze(0),
        "count": np.array([arr.shape[0]], dtype=np.int64),
    }
    for k, qv in zip(QUANTILE_KEYS, q_vals, strict=True):
        stats[k] = qv.squeeze(0)
    return stats


def decode_and_verify_video_pyav(
    path: Path,
    *,
    expected_fps: float,
    sample_indices_for_stats: set[int] | None = None,
) -> tuple[VideoInfo, np.ndarray, float, dict[str, np.ndarray] | None]:
    """Decode entire MP4 via PyAV, returning (VideoInfo, first_rgb, max_pts_err_s, optional_stats)."""
    count = 0
    first_frame: np.ndarray | None = None
    pts_list: list[float] = []
    sampled_chw: list[np.ndarray] = []

    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in {path}")
        stream = container.streams.video[0]
        width, height = int(stream.width), int(stream.height)
        fps = float(_stream_fps(stream))
        codec_name = str(stream.codec_context.name or "")
        pix_fmt = str(stream.codec_context.pix_fmt or "")
        has_audio = bool(container.streams.audio)
        time_base = float(stream.time_base) if stream.time_base is not None else 0.0

        for frame in container.decode(stream):
            rgb = frame.to_ndarray(format="rgb24")
            if first_frame is None:
                first_frame = rgb
            if frame.pts is not None and time_base > 0.0:
                pts_list.append(float(frame.pts) * time_base)
            else:
                pts_list.append(float(count) / max(fps, 1e-6))
            if sample_indices_for_stats is not None and count in sample_indices_for_stats:
                sampled_chw.append(_auto_downsample_chw(np.transpose(rgb, (2, 0, 1))))
            count += 1

    if first_frame is None or count == 0:
        raise ValueError(f"No decodable video frames: {path}")

    expected_ts = (np.arange(count, dtype=np.float32) / np.float32(expected_fps)).astype(np.float64)
    pts_arr = np.asarray(pts_list, dtype=np.float64)
    max_pts_err_s = float(np.max(np.abs(expected_ts - pts_arr)))

    info = VideoInfo(
        width=width,
        height=height,
        fps=fps,
        frames=count,
        duration=float(count) / fps if fps > 0 else 0.0,
        codec=codec_name,
        pix_fmt=pix_fmt,
        has_audio=has_audio,
    )
    vid_stats = _compute_image_channel_stats(sampled_chw) if sampled_chw else None
    return info, first_frame, max_pts_err_s, vid_stats


def encode_bgr_frames_h264(
    frames_bgr: list[np.ndarray],
    dst: Path,
    *,
    camera_key: str,
    fps: int,
    target_size: tuple[int, int] | None = (224, 224),
    resize_mode: str = "pad",
    codec: str = DEFAULT_CODEC,
    crf: int = DEFAULT_CRF,
    gop: int = DEFAULT_GOP,
    preset: str = DEFAULT_PRESET,
    verify: bool = True,
    max_first_frame_mae: float = DEFAULT_MAX_FIRST_FRAME_MAE,
    max_pad_mean: float = DEFAULT_MAX_PAD_MEAN,
    pts_tolerance_s: float = DEFAULT_PTS_TOLERANCE_S,
) -> tuple[VideoInfo, dict[str, np.ndarray], VideoVerifyMetrics]:
    """Atomically encode BGR frames -> H.264 MP4, verify stream/PTS/MAE/padding, and compute stats."""
    if not frames_bgr:
        raise ValueError(f"No frames to encode for {camera_key}")

    input_height, input_width = frames_bgr[0].shape[:2]
    if target_size is not None:
        out_w, out_h = target_size
    else:
        out_w, out_h = input_width, input_height

    dst.parent.mkdir(parents=True, exist_ok=True)
    temporary = dst.with_name(dst.stem + ".encoding" + dst.suffix)
    rate = Fraction(int(fps), 1)
    n_frames = len(frames_bgr)
    sample_idx_list = _sample_video_indices(n_frames)
    sample_set = set(sample_idx_list)

    options: dict[str, str] = {"crf": str(crf), "g": str(gop)}
    if preset and codec in ("libx264", "h264"):
        options["preset"] = str(preset)

    first_expected: np.ndarray | None = None
    fallback_sampled_chw: list[np.ndarray] = []
    encoded_count = 0

    logging.getLogger("libav").setLevel(av.logging.WARNING)
    try:
        with av.open(str(temporary), mode="w") as output:
            stream = output.add_stream(codec, rate=rate, options=options)
            stream.pix_fmt = "yuv420p"
            stream.width = out_w
            stream.height = out_h

            for idx, img_bgr in enumerate(frames_bgr):
                if target_size is not None:
                    img_bgr = resize_bgr_frame(img_bgr, target_size, resize_mode)
                rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])
                if first_expected is None:
                    first_expected = rgb
                if not verify and idx in sample_set:
                    fallback_sampled_chw.append(_auto_downsample_chw(np.transpose(rgb, (2, 0, 1))))
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                for packet in stream.encode(frame):
                    output.mux(packet)
                encoded_count += 1

            for packet in stream.encode():
                output.mux(packet)

        if encoded_count == 0 or first_expected is None:
            raise ValueError(f"Encoded 0 frames for {camera_key}")
        temporary.replace(dst)

        if verify:
            info, first_output, max_pts_err_s, vid_stats = decode_and_verify_video_pyav(
                dst,
                expected_fps=float(fps),
                sample_indices_for_stats=sample_set,
            )
            if (info.width, info.height) != (out_w, out_h):
                raise ValueError(
                    f"Wrong output size for {camera_key}: {info.width}x{info.height}, expected {out_w}x{out_h}"
                )
            if info.frames != encoded_count:
                raise ValueError(f"Frame count mismatch for {camera_key}: {encoded_count} -> {info.frames}")
            if not math.isclose(info.fps, float(fps), rel_tol=0.0, abs_tol=1e-3):
                raise ValueError(f"FPS mismatch for {camera_key}: {fps} -> {info.fps}")
            expected_codec_name = "h264" if codec in ("libx264", "h264") else None
            if (expected_codec_name and info.codec != expected_codec_name) or info.pix_fmt != "yuv420p" or info.has_audio:
                raise ValueError(
                    f"Unexpected video stream properties for {camera_key}: "
                    f"codec={info.codec}, pix_fmt={info.pix_fmt}, has_audio={info.has_audio}"
                )
            if max_pts_err_s >= pts_tolerance_s:
                raise ValueError(
                    f"PTS tolerance exceeded for {camera_key}: max_err={max_pts_err_s:.6f}s >= {pts_tolerance_s}"
                )

            first_frame_mae = float(
                np.abs(first_output.astype(np.float32) - first_expected.astype(np.float32)).mean()
            )
            if first_frame_mae > max_first_frame_mae:
                raise ValueError(
                    f"First-frame MAE too high for {camera_key}: "
                    f"{first_frame_mae:.3f} > {max_first_frame_mae:.3f}"
                )

            if target_size is not None and resize_mode == "pad":
                pad_regions = _padding_slices(input_width, input_height, out_w, out_h)
                if pad_regions:
                    pad_pixels = np.concatenate(
                        [first_output[rows, cols].reshape(-1, 3) for rows, cols in pad_regions],
                        axis=0,
                    )
                    pad_mean = float(pad_pixels.mean())
                    if pad_mean > max_pad_mean:
                        raise ValueError(
                            f"Letterbox padding is not black enough for {camera_key}: "
                            f"mean={pad_mean:.3f} > {max_pad_mean:.3f}"
                        )
                else:
                    pad_mean = 0.0
            else:
                pad_mean = 0.0
            assert vid_stats is not None
        else:
            info = probe_video_pyav(dst)
            vid_stats = _compute_image_channel_stats(fallback_sampled_chw)
            first_frame_mae = float("nan")
            pad_mean = float("nan")
            max_pts_err_s = float("nan")

        metrics = VideoVerifyMetrics(
            camera_key=camera_key,
            input_width=input_width,
            input_height=input_height,
            output_width=info.width,
            output_height=info.height,
            output_fps=info.fps,
            output_frames=info.frames,
            first_frame_mae=first_frame_mae,
            pad_mean=pad_mean,
            max_pts_err_s=max_pts_err_s,
        )
        return info, vid_stats, metrics
    except Exception:
        temporary.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)
        raise


def video_feature_info(info: VideoInfo, fps: int) -> dict[str, Any]:
    return {
        "video.height": info.height,
        "video.width": info.width,
        "video.codec": info.codec,
        "video.pix_fmt": info.pix_fmt,
        "video.is_depth_map": False,
        "video.fps": fps,
        "video.channels": 3,
        "has_audio": info.has_audio,
    }


# ============================================================================
# 4. MCAP Protobuf Decoding & Strict Extraction
# ============================================================================
def _build_descriptor_pool(reader: SeekingReader) -> descriptor_pool.DescriptorPool:
    summary = reader.get_summary()
    pool = descriptor_pool.DescriptorPool()
    all_fds: list[descriptor_pb2.FileDescriptorProto] = []
    for schema in summary.schemas.values():
        if schema.encoding == "protobuf":
            fds = descriptor_pb2.FileDescriptorSet()
            fds.ParseFromString(schema.data)
            all_fds.extend(fds.file)

    added: set[str] = set()

    def _add_fd(fd: descriptor_pb2.FileDescriptorProto) -> None:
        if fd.name in added:
            return
        for dep in fd.dependency:
            if dep not in added:
                for other in all_fds:
                    if other.name == dep:
                        _add_fd(other)
                        break
        pool.Add(fd)
        added.add(fd.name)

    for fd in all_fds:
        _add_fd(fd)
    return pool


def _decode_message(
    message_data: bytes,
    schema_name: str,
    pool: descriptor_pool.DescriptorPool,
    cls_cache: dict[str, Any] | None = None,
) -> Any:
    if cls_cache is not None and schema_name in cls_cache:
        cls = cls_cache[schema_name]
    else:
        desc = pool.FindMessageTypeByName(schema_name)
        cls = message_factory.GetMessageClass(desc)
        if cls_cache is not None:
            cls_cache[schema_name] = cls
    msg = cls()
    msg.ParseFromString(message_data)
    return msg


def _msg_ts_ns(msg: Any) -> int:
    ts = msg.header.timestamp
    return int(ts.sec) * 1_000_000_000 + int(ts.nanosec)


def _decode_image_bgr_from_message(msg: Any) -> np.ndarray | None:
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _extract_sensor_vector(msg: Any, groups: list[str]) -> tuple[int, np.ndarray] | None:
    """Extract strictly validated joint sensor vector; return None if any required group is missing."""
    ts = _msg_ts_ns(msg)
    source_map = msg.joint_sensor_map
    parts: list[list[float]] = []
    for group_name in groups:
        expected_dim = DIM_MAPS[group_name]
        if group_name not in source_map:
            return None
        positions = list(source_map[group_name].position)
        if len(positions) != expected_dim:
            raise ValueError(
                f"Sensor group {group_name!r} dim={len(positions)}, expected {expected_dim}"
            )
        parts.append(positions)
    flat = [v for sub in parts for v in sub]
    return ts, np.asarray(flat, dtype=np.float32)


def _extract_target_group_vector(
    msg: Any,
    requested_groups: set[str],
) -> list[tuple[int, str, np.ndarray]]:
    ts = _msg_ts_ns(msg)
    source_map = msg.target_group_trajectory_map
    if not source_map:
        if SKIP_EMPTY_ACTIONS:
            return []
        raise ValueError("Target message has empty target_group_trajectory_map")
    results: list[tuple[int, str, np.ndarray]] = []
    for group_name in source_map:
        if group_name not in requested_groups:
            continue
        group = source_map[group_name]
        if not group.group_commands:
            raise ValueError(f"Target group {group_name} has no group_commands")
        cmd = group.group_commands[0]
        positions = [jc.position for jc in cmd.joint_commands]
        expected_dim = DIM_MAPS[group_name]
        if len(positions) != expected_dim:
            raise ValueError(
                f"Target group {group_name!r} dim={len(positions)}, expected {expected_dim}"
            )
        results.append((ts, group_name, np.asarray(positions, dtype=np.float32)))
    return results


def extract_all_from_mcap(
    mcap_path: Path,
    joint_groups: list[str],
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[str, list[tuple[int, np.ndarray]]],
    dict[str, list[tuple[int, np.ndarray]]],
    dict[str, tuple[int, int]],
]:
    sensor_ts_list: list[int] = []
    sensor_pos_list: list[np.ndarray] = []
    dropped_sensor_msgs = 0
    total_sensor_msgs = 0
    target_group_data: dict[str, list[tuple[int, np.ndarray]]] = {g: [] for g in joint_groups}
    camera_data: dict[str, list[tuple[int, np.ndarray]]] = {t: [] for t in CAMERA_TOPICS.values()}
    camera_shapes: dict[str, tuple[int, int]] = {}
    requested_groups = set(joint_groups)
    topic_set = set(SENSOR_TOPICS) | set(ACTION_TOPICS) | set(CAMERA_TOPICS.values())

    with open(mcap_path, "rb") as f:
        reader = SeekingReader(f, record_size_limit=None)
        pool = _build_descriptor_pool(reader)
        schemas_by_id = dict(reader.get_summary().schemas.items())
        cls_cache: dict[str, Any] = {}

        for _, channel, message in reader.iter_messages(topics=list(topic_set)):
            schema = schemas_by_id[channel.schema_id]
            msg = _decode_message(message.data, schema.name, pool, cls_cache)
            topic = channel.topic
            if topic in SENSOR_TOPICS:
                total_sensor_msgs += 1
                extracted = _extract_sensor_vector(msg, joint_groups)
                if extracted is None:
                    dropped_sensor_msgs += 1
                    continue
                ts, pos = extracted
                sensor_ts_list.append(ts)
                sensor_pos_list.append(pos)
            elif topic in ACTION_TOPICS:
                for ts, group_name, pos in _extract_target_group_vector(msg, requested_groups):
                    target_group_data[group_name].append((ts, pos))
            elif topic in camera_data:
                img = _decode_image_bgr_from_message(msg)
                if img is None:
                    logging.warning("Failed to decode frame for %s in %s", topic, mcap_path.name)
                    continue
                if topic not in camera_shapes:
                    camera_shapes[topic] = (int(img.shape[0]), int(img.shape[1]))
                try:
                    ts = _msg_ts_ns(msg)
                except Exception:
                    ts = message.log_time
                camera_data[topic].append((ts, img))

    if dropped_sensor_msgs > 0:
        logging.warning(
            "[%s] Dropped %d/%d incomplete sensor messages missing required joint groups",
            mcap_path.name,
            dropped_sensor_msgs,
            total_sensor_msgs,
        )
    if not sensor_ts_list:
        raise ValueError(f"No valid sensor messages found in {mcap_path}")
    if total_sensor_msgs > 0 and dropped_sensor_msgs / total_sensor_msgs > 0.5:
        raise ValueError(
            f"Over 50% sensor messages incomplete ({dropped_sensor_msgs}/{total_sensor_msgs}) in {mcap_path}"
        )

    return (
        np.asarray(sensor_ts_list, dtype=np.int64),
        np.stack(sensor_pos_list),
        target_group_data,
        camera_data,
        camera_shapes,
    )


# ============================================================================
# 5. Timestamp Grid, Causal ZOH Alignment, Dropout Detection & Segmentation
# ============================================================================
def nearest_indices_by_timestamp(target_ts: np.ndarray, source_ts: np.ndarray) -> np.ndarray:
    target_ts = np.asarray(target_ts, dtype=np.int64)
    source_ts = np.asarray(source_ts, dtype=np.int64)
    if len(source_ts) == 0:
        raise ValueError("Cannot align against an empty timestamp array.")
    order = np.argsort(source_ts, kind="stable")
    sorted_ts = source_ts[order]
    insertion = np.searchsorted(sorted_ts, target_ts, side="left")
    right = np.clip(insertion, 0, len(sorted_ts) - 1)
    left = np.clip(insertion - 1, 0, len(sorted_ts) - 1)
    left_delta = np.abs(target_ts - sorted_ts[left])
    right_delta = np.abs(sorted_ts[right] - target_ts)
    return order[np.where(left_delta <= right_delta, left, right)]


def causal_zoh_indices_and_deltas(
    target_ts: np.ndarray,
    source_ts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Causal Zero-Order Hold: for each grid timestamp t, pick latest source_ts <= t.

    Frames before the first command (`t < source_ts[0]`) return index `-1` (hold state).
    """
    target_ts = np.asarray(target_ts, dtype=np.int64)
    source_ts = np.asarray(source_ts, dtype=np.int64)
    if source_ts.ndim != 1 or target_ts.ndim != 1:
        raise ValueError("Timestamp arrays must be one-dimensional.")
    if len(source_ts) == 0:
        raise ValueError("Cannot align against an empty timestamp array.")

    order = np.argsort(source_ts, kind="stable")
    sorted_ts = source_ts[order]
    pos = np.searchsorted(sorted_ts, target_ts, side="right") - 1

    indices = np.full(len(target_ts), -1, dtype=np.int64)
    ages_ns = np.full(len(target_ts), -1, dtype=np.int64)
    active = pos >= 0
    if active.any():
        indices[active] = order[pos[active]]
        ages_ns[active] = target_ts[active] - sorted_ts[pos[active]]
    return indices, ages_ns, int(sorted_ts[0])


def build_group_slices(joint_groups: list[str]) -> dict[str, slice]:
    slices: dict[str, slice] = {}
    offset = 0
    for group_name in joint_groups:
        dim = DIM_MAPS[group_name]
        slices[group_name] = slice(offset, offset + dim)
        offset += dim
    return slices


def build_uniform_timestamp_grid(stream_timestamps: list[np.ndarray], fps: int) -> np.ndarray:
    """Build a uniform constant-FPS int64 nanosecond grid over the overlapping stream window."""
    if fps <= 0:
        raise ValueError(f"fps must be positive for uniform resampling, got {fps}")
    starts = [int(np.min(ts)) for ts in stream_timestamps if len(ts) > 0]
    ends = [int(np.max(ts)) for ts in stream_timestamps if len(ts) > 0]
    if not starts:
        raise ValueError("Cannot build timestamp grid from empty streams.")
    t_start = max(starts)
    t_end = min(ends)
    if t_end <= t_start:
        raise ValueError(f"Streams have no overlapping time window (t_start={t_start}, t_end={t_end}).")
    step_ns = int(round(1_000_000_000.0 / float(fps)))
    return np.arange(t_start, t_end + 1, step_ns, dtype=np.int64)


def validate_and_segment_action_alignment(
    master_ts: np.ndarray,
    sensor_ts: np.ndarray,
    sensor_arr: np.ndarray,
    target_group_data: dict[str, list[tuple[int, np.ndarray]]],
    joint_groups: list[str],
    *,
    episode_name: str,
    arm_tolerance_ms: float = DEFAULT_ARM_TOLERANCE_MS,
    arm_startup_grace_s: float = DEFAULT_ARM_STARTUP_GRACE_S,
    arm_startup_max_gap_ms: float = DEFAULT_ARM_STARTUP_MAX_GAP_MS,
    hold_eps_rad: float = DEFAULT_HOLD_EPS_RAD,
    allow_missing_groups: bool = True,
    trim_leading_idle: bool = True,
    split_on_interior_gaps: bool = True,
    min_segment_frames: int = DEFAULT_MIN_FRAMES,
) -> tuple[list[slice], dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """Causal ZOH alignment + sequential-bimanual validation + gap segmentation.

    Preserves conveyor waiting frames where target streams are already publishing stationary
    commands; only trims frames strictly before `min(first_target_ts)` when `trim_leading_idle`
    is enabled. Rescues intentional end-of-episode holds (where `max|state - last_target| <=
    hold_eps_rad`) and splits on true mid-trajectory target dropouts.
    """
    num_frames = len(master_ts)
    if num_frames == 0:
        raise ValueError(f"Timestamp grid has 0 frames in {episode_name}")

    group_slices = build_group_slices(joint_groups)
    sensor_nearest = nearest_indices_by_timestamp(master_ts, sensor_ts)
    state_full = sensor_arr[sensor_nearest].astype(np.float32, copy=False)
    sensor_max_err_ms = float(np.max(np.abs(master_ts - sensor_ts[sensor_nearest])) / 1e6)

    gated_group_set = {"left_arm", "right_arm"}
    gate_ns = float(arm_tolerance_ms) * 1_000_000.0
    startup_grace_ns = int(round(float(arm_startup_grace_s) * 1_000_000_000.0))
    startup_max_gap_ns = float(arm_startup_max_gap_ms) * 1_000_000.0

    causal_by_group: dict[str, np.ndarray] = {}
    ages_by_group: dict[str, np.ndarray] = {}
    first_ts_by_group: dict[str, int] = {}
    missing_groups: list[str] = []
    exempt_groups: list[str] = []

    for group_name in joint_groups:
        items = target_group_data.get(group_name, [])
        if not items:
            if allow_missing_groups:
                missing_groups.append(group_name)
                continue
            raise ValueError(f"Action group {group_name!r} has no target messages in {episode_name}")

        group_ts = np.asarray([ts for ts, _ in items], dtype=np.int64)
        c_idx, ages_ns, first_ts = causal_zoh_indices_and_deltas(master_ts, group_ts)
        causal_by_group[group_name] = c_idx
        ages_by_group[group_name] = ages_ns
        first_ts_by_group[group_name] = first_ts

    active_arm_first_ts = [
        first_ts_by_group[g] for g in joint_groups if g in gated_group_set and g in first_ts_by_group
    ]
    earliest_arm_start_ts: int | None = min(active_arm_first_ts) if active_arm_first_ts else None

    combined_valid = np.ones(num_frames, dtype=bool)
    leading_idle_trimmed = 0
    if trim_leading_idle and earliest_arm_start_ts is not None:
        before_any_arm = master_ts < earliest_arm_start_ts
        leading_idle_trimmed = int(before_any_arm.sum())
        combined_valid &= ~before_any_arm

    hold_rescued_by_group: dict[str, int] = {}
    startup_rescued_by_group: dict[str, int] = {}
    pre_start_frames_by_group: dict[str, int] = {}
    max_drift_by_group: dict[str, float | None] = {}

    for group_name in joint_groups:
        if group_name not in causal_by_group:
            continue
        if group_name not in gated_group_set or arm_tolerance_ms <= 0:
            exempt_groups.append(group_name)
            continue

        items = target_group_data[group_name]
        group_pos = np.stack([pos for _, pos in items])
        c_idx = causal_by_group[group_name]
        ages_ns = ages_by_group[group_name]
        first_ts = first_ts_by_group[group_name]
        sl = group_slices[group_name]

        pre_mask = (c_idx < 0) & combined_valid
        pre_start_frames_by_group[group_name] = int(pre_mask.sum())
        if pre_mask.any():
            pre_indices = np.flatnonzero(pre_mask)
            ref_pose = state_full[pre_indices[0], sl]
            pre_drift = np.max(np.abs(state_full[pre_indices, sl] - ref_pose), axis=1)
            bad_pre = pre_indices[pre_drift > hold_eps_rad]
            if len(bad_pre) > 0:
                combined_valid[bad_pre] = False

        active_mask = c_idx >= 0
        over = active_mask & (ages_ns > gate_ns)
        if over.any():
            over_idx = np.flatnonzero(over)
            drift_over = np.max(
                np.abs(state_full[over_idx, sl] - group_pos[c_idx[over_idx]]), axis=1
            )
            hold_ok_local = drift_over <= hold_eps_rad
            startup_ok_local = (
                (master_ts[over_idx] <= first_ts + startup_grace_ns)
                & (ages_ns[over_idx] <= startup_max_gap_ns)
            )
            hold_rescued_by_group[group_name] = int(hold_ok_local.sum())
            startup_rescued_by_group[group_name] = int((~hold_ok_local & startup_ok_local).sum())
            max_drift_by_group[group_name] = float(np.max(drift_over))

            bad_over = over_idx[~(hold_ok_local | startup_ok_local)]
            if len(bad_over) > 0:
                combined_valid[bad_over] = False
        else:
            hold_rescued_by_group[group_name] = 0
            startup_rescued_by_group[group_name] = 0
            max_drift_by_group[group_name] = None

    valid_idx = np.flatnonzero(combined_valid)
    if len(valid_idx) == 0:
        raise ValueError(f"All {num_frames} frames failed action alignment in {episode_name}")

    boundaries = np.flatnonzero(np.diff(valid_idx) > 1)
    starts = np.r_[valid_idx[0], valid_idx[boundaries + 1]]
    ends = np.r_[valid_idx[boundaries], valid_idx[-1]]
    raw_runs = [slice(int(s), int(e) + 1) for s, e in zip(starts, ends)]

    if len(raw_runs) > 1 and not split_on_interior_gaps:
        raise ValueError(
            f"Interior action gap detected in {episode_name}: {len(raw_runs)} discontinuous runs"
        )

    kept_runs = [r for r in raw_runs if (r.stop - r.start) >= min_segment_frames]
    if not kept_runs:
        longest = max(r.stop - r.start for r in raw_runs)
        raise ValueError(
            f"No valid segment >= {min_segment_frames} frames in {episode_name} (longest={longest})"
        )

    kept_mask = np.zeros(num_frames, dtype=bool)
    for run in kept_runs:
        kept_mask[run] = True

    group_report: dict[str, dict[str, Any]] = {}
    for group_name, ages_ns in ages_by_group.items():
        active_kept = kept_mask & (causal_by_group[group_name] >= 0)
        if active_kept.any():
            kept_ms = ages_ns[active_kept].astype(np.float64) / 1_000_000.0
            p95_ms = round(float(np.percentile(kept_ms, 95)), 3)
            max_ms = round(float(np.max(kept_ms)), 3)
        else:
            p95_ms = 0.0
            max_ms = 0.0
        group_report[group_name] = {
            "gate_ms": arm_tolerance_ms if group_name in gated_group_set else None,
            "pre_start_state_hold_frames": pre_start_frames_by_group.get(group_name, 0),
            "kept_active_p95_ms": p95_ms,
            "kept_active_max_ms": max_ms,
            "startup_rescued_frames": startup_rescued_by_group.get(group_name, 0),
            "hold_rescued_frames": hold_rescued_by_group.get(group_name, 0),
            "max_drift_on_over_rad": max_drift_by_group.get(group_name),
        }

    report: dict[str, Any] = {
        "alignment_mode": "causal_zoh_sequential_bimanual",
        "sensor_max_sync_error_ms": round(sensor_max_err_ms, 3),
        "original_frames": int(num_frames),
        "kept_frames": int(kept_mask.sum()),
        "dropped_frames": int(num_frames - kept_mask.sum()),
        "leading_idle_trimmed": leading_idle_trimmed,
        "num_raw_runs": len(raw_runs),
        "num_kept_runs": len(kept_runs),
        "runs": [{"start": r.start, "stop": r.stop, "frames": r.stop - r.start} for r in kept_runs],
        "missing_groups_filled_from_state": missing_groups,
        "exempt_groups": exempt_groups,
        "action_groups": group_report,
    }
    return kept_runs, causal_by_group, state_full, report


# ============================================================================
# 6. Episode Conversion Worker (MCAP -> Validated Segments + Staged Videos)
# ============================================================================
def _convert_episode_worker(job: dict[str, Any]) -> dict[str, Any]:
    """Process one MCAP in a worker process and write temporary staged MP4 files."""
    mcap_path = Path(job["mcap_path"])
    staging_dir = Path(job["staging_dir"])
    ident = str(job["ident"])
    fps_override = job["fps_override"]
    min_frames = int(job["min_frames"])
    target_size = tuple(job["target_size"]) if job["target_size"] is not None else None
    resize_mode = str(job["resize_mode"])
    resample_uniform = bool(job["resample_uniform"])

    try:
        sensor_ts, sensor_arr, target_group_data, camera_data, camera_shapes = extract_all_from_mcap(
            mcap_path, JOINT_GROUPS
        )
        topic_to_key = {v: k for k, v in CAMERA_TOPICS.items()}
        cam_ts_map: dict[str, np.ndarray] = {}
        measured_camera_fps: dict[str, float] = {}
        for topic, key in topic_to_key.items():
            items = camera_data.get(topic) or []
            if not items:
                raise ValueError(f"No frames for camera topic {topic}")
            ts_arr = np.asarray([ts for ts, _ in items], dtype=np.int64)
            cam_ts_map[topic] = ts_arr
            if len(ts_arr) > 1:
                dur_s = max(1e-6, (int(ts_arr[-1]) - int(ts_arr[0])) / 1e9)
                measured_camera_fps[key] = float((len(ts_arr) - 1) / dur_s)
            else:
                measured_camera_fps[key] = float(DEFAULT_FPS)

        head_topic = CAMERA_TOPICS["observation.images.head_right"]
        head_fps = measured_camera_fps["observation.images.head_right"]
        fps = int(fps_override) if fps_override is not None else max(1, int(round(head_fps)))

        for key, cam_fps in measured_camera_fps.items():
            if abs(cam_fps - fps) > 1.5:
                logging.warning(
                    "[%s] Camera %s measured_fps=%.2f deviates from target fps=%d",
                    mcap_path.name,
                    key,
                    cam_fps,
                    fps,
                )

        if resample_uniform:
            master_ts = build_uniform_timestamp_grid([sensor_ts, *cam_ts_map.values()], fps=fps)
        else:
            master_ts = cam_ts_map[head_topic]

        kept_runs, causal_by_group, state_full, align_report = validate_and_segment_action_alignment(
            master_ts=master_ts,
            sensor_ts=sensor_ts,
            sensor_arr=sensor_arr,
            target_group_data=target_group_data,
            joint_groups=JOINT_GROUPS,
            episode_name=mcap_path.name,
            arm_tolerance_ms=float(job["arm_tolerance_ms"]),
            arm_startup_grace_s=float(job["arm_startup_grace_s"]),
            arm_startup_max_gap_ms=float(job["arm_startup_max_gap_ms"]),
            hold_eps_rad=float(job["hold_eps_rad"]),
            allow_missing_groups=bool(job["allow_missing_groups"]),
            trim_leading_idle=bool(job["trim_leading_idle"]),
            split_on_interior_gaps=bool(job["split_on_interior_gaps"]),
            min_segment_frames=min_frames,
        )

        group_slices = build_group_slices(JOINT_GROUPS)
        target_pos_by_group = {
            g: np.stack([pos for _, pos in target_group_data[g]])
            for g in JOINT_GROUPS
            if target_group_data.get(g)
        }

        safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", ident)
        segments_out: list[dict[str, Any]] = []
        created_tmp_videos: list[Path] = []

        try:
            for seg_idx, run in enumerate(kept_runs):
                seg_master_ts = master_ts[run]
                num_frames = len(seg_master_ts)
                state = state_full[run].copy()
                action = state.copy()

                for group_name, group_pos in target_pos_by_group.items():
                    seg_cidx = causal_by_group[group_name][run]
                    active = seg_cidx >= 0
                    if active.any():
                        action[active, group_slices[group_name]] = group_pos[seg_cidx[active]]

                tmp_videos: dict[str, str] = {}
                video_infos: dict[str, dict[str, Any]] = {}
                video_stats: dict[str, dict[str, list[Any]]] = {}
                verify_metrics: list[dict[str, Any]] = []
                letterbox_infos: dict[str, dict[str, int]] = {}
                cam_sync_err_ms: dict[str, float] = {}

                for topic, key in topic_to_key.items():
                    items = camera_data[topic]
                    topic_ts = cam_ts_map[topic]
                    frames = [img for _, img in items]
                    if not resample_uniform and topic == head_topic:
                        aligned_frames = frames[run]
                        cam_sync_err_ms[key] = 0.0
                    else:
                        cam_nearest = nearest_indices_by_timestamp(seg_master_ts, topic_ts)
                        aligned_frames = [frames[i] for i in cam_nearest]
                        cam_sync_err_ms[key] = round(
                            float(np.max(np.abs(seg_master_ts - topic_ts[cam_nearest])) / 1e6), 3
                        )

                    if target_size is not None and resize_mode == "pad" and topic in camera_shapes:
                        src_h, src_w = camera_shapes[topic]
                        letterbox_infos[key] = compute_letterbox_padding(
                            src_w, src_h, target_size[0], target_size[1]
                        )

                    if not bool(job["skip_videos"]):
                        tmp_dst = staging_dir / f"{safe_tag}__seg{seg_idx}__{key.replace('.', '_')}.mp4"
                        v_info, v_stats, v_metrics = encode_bgr_frames_h264(
                            aligned_frames,
                            tmp_dst,
                            camera_key=key,
                            fps=fps,
                            target_size=target_size,
                            resize_mode=resize_mode,
                            codec=str(job["codec"]),
                            crf=int(job["crf"]),
                            gop=int(job["gop"]),
                            preset=str(job["preset"]),
                            verify=not bool(job["no_video_verify"]),
                            max_first_frame_mae=float(job["max_first_frame_mae"]),
                            max_pad_mean=float(job["max_pad_mean"]),
                            pts_tolerance_s=float(job["pts_tolerance_s"]),
                        )
                        created_tmp_videos.append(tmp_dst)
                        tmp_videos[key] = str(tmp_dst)
                        video_infos[key] = asdict(v_info)
                        video_stats[key] = {sk: np.asarray(sv).tolist() for sk, sv in v_stats.items()}
                        verify_metrics.append(asdict(v_metrics))

                frame_index = np.arange(num_frames, dtype=np.int64)
                timestamp_sec = (frame_index / float(fps)).astype(np.float32)

                segments_out.append(
                    {
                        "seg_idx": seg_idx,
                        "state": state,
                        "action": action,
                        "timestamp": timestamp_sec,
                        "frame_index": frame_index,
                        "num_frames": num_frames,
                        "fps": fps,
                        "tmp_videos": tmp_videos,
                        "video_infos": video_infos,
                        "video_stats": video_stats,
                        "verify_metrics": verify_metrics,
                        "camera_shapes": {
                            topic_to_key[t]: list(hw) for t, hw in camera_shapes.items()
                        },
                        "letterbox_infos": letterbox_infos,
                        "timing": {
                            "resample_uniform": resample_uniform,
                            "target_fps": int(fps),
                            "measured_camera_fps": {
                                k: round(v, 3) for k, v in measured_camera_fps.items()
                            },
                            "camera_max_sync_error_ms": cam_sync_err_ms,
                        },
                        "alignment": align_report,
                    }
                )
        except Exception:
            for p_tmp in created_tmp_videos:
                p_tmp.unlink(missing_ok=True)
            raise

        return {
            "ok": True,
            "ident": ident,
            "task_name": job["task_name"],
            "segments": segments_out,
        }
    except Exception as exc:
        return {"ok": False, "ident": ident, "error": str(exc)}


# ============================================================================
# 7. Parquet Writing, Features, RunningQuantileStats & VLASH Relative Stats
# ============================================================================
def fixed_size_list(values: np.ndarray, dtype: pa.DataType) -> pa.FixedSizeListArray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"Expected 2-D array for fixed size list column, got {array.shape}")
    flat = pa.array(array.reshape(-1), type=dtype)
    return pa.FixedSizeListArray.from_arrays(flat, list_size=array.shape[1])


def write_episode_parquet(
    path: Path,
    *,
    state: np.ndarray,
    action: np.ndarray,
    subtask_index: np.ndarray,
    timestamp: np.ndarray,
    frame_index: np.ndarray,
    episode_index: np.ndarray,
    index: np.ndarray,
    task_index: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    table = pa.table(
        {
            OBS_STATE: fixed_size_list(state.astype(np.float32, copy=False), pa.float32()),
            ACTION: fixed_size_list(action.astype(np.float32, copy=False), pa.float32()),
            "subtask_index": pa.array(subtask_index.astype(np.int64, copy=False), type=pa.int64()),
            "timestamp": pa.array(timestamp.astype(np.float32, copy=False), type=pa.float32()),
            "frame_index": pa.array(frame_index.astype(np.int64, copy=False), type=pa.int64()),
            "episode_index": pa.array(episode_index.astype(np.int64, copy=False), type=pa.int64()),
            "index": pa.array(index.astype(np.int64, copy=False), type=pa.int64()),
            "task_index": pa.array(task_index.astype(np.int64, copy=False), type=pa.int64()),
        }
    )
    pq.write_table(table, tmp_path, compression="snappy", use_dictionary=True)
    os.replace(tmp_path, path)


def make_lerobot_features(
    height: int,
    width: int,
    action_dim: int,
    state_dim: int,
    video_infos: dict[str, VideoInfo] | None,
    fps: int,
) -> dict[str, dict[str, Any]]:
    names = ARM_STATE_ACTION_NAMES if state_dim == 16 else [f"joint_{i:02d}" for i in range(state_dim)]
    features: dict[str, dict[str, Any]] = {
        OBS_STATE: {"dtype": "float32", "shape": [state_dim], "names": names},
        ACTION: {"dtype": "float32", "shape": [action_dim], "names": names},
        "subtask_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for video_key in CAMERA_TOPICS:
        info = video_infos.get(video_key) if video_infos else None
        cam_h = info.height if info else height
        cam_w = info.width if info else width
        features[video_key] = {
            "dtype": "video",
            "shape": [cam_h, cam_w, 3],
            "names": ["height", "width", "channels"],
            "info": video_feature_info(info, fps) if info else None,
        }
    for k, v in DEFAULT_FEATURES.items():
        features[k] = {"dtype": v["dtype"], "shape": list(v["shape"]), "names": v["names"]}
    return features


class RunningQuantileStats:
    """Streaming float64 quantile & moment accumulator matching lerobot/datasets/compute_stats.py."""

    def __init__(self, quantile_list: list[float] | None = None, num_quantile_bins: int = 5000):
        self._count = 0
        self._mean: np.ndarray | None = None
        self._mean_of_squares: np.ndarray | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._histograms: list[np.ndarray] | None = None
        self._bin_edges: list[np.ndarray] | None = None
        self._num_quantile_bins = num_quantile_bins
        self._quantile_list = quantile_list if quantile_list is not None else QUANTILES
        self._quantile_keys = [f"q{int(q * 100):02d}" for q in self._quantile_list]

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64).reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if num_elements == 0:
            return
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            assert self._mean is not None and self._min is not None and self._max is not None
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = bool(np.any(new_max > self._max))
            min_changed = bool(np.any(new_min < self._min))
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)
            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements
        batch_mean = np.mean(batch, axis=0)
        batch_mean_sq = np.mean(batch**2, axis=0)
        assert self._mean is not None and self._mean_of_squares is not None
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_sq - self._mean_of_squares) * (num_elements / self._count)
        self._update_histograms(batch)

    def get_statistics(self) -> dict[str, np.ndarray]:
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")
        assert self._mean is not None and self._mean_of_squares is not None
        assert self._min is not None and self._max is not None
        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0.0, variance))
        stats: dict[str, np.ndarray] = {
            "min": self._min.copy(),
            "max": self._max.copy(),
            "mean": self._mean.copy(),
            "std": stddev,
            "count": np.array([self._count], dtype=np.int64),
        }
        for key, q_val in zip(self._quantile_keys, self._compute_quantiles(), strict=True):
            stats[key] = q_val
        return stats

    def _adjust_histograms(self) -> None:
        assert self._histograms is not None and self._bin_edges is not None
        assert self._min is not None and self._max is not None
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            old_hist = self._histograms[i]
            padding = max(1e-10, (self._max[i] - self._min[i]) * 1e-10)
            new_edges = np.linspace(
                self._min[i] - padding, self._max[i] + padding, self._num_quantile_bins + 1
            )
            old_centers = (old_edges[:-1] + old_edges[1:]) / 2.0
            new_hist = np.zeros(self._num_quantile_bins)
            for old_center, count in zip(old_centers, old_hist, strict=False):
                if count > 0:
                    bin_idx = int(np.searchsorted(new_edges, old_center) - 1)
                    bin_idx = max(0, min(bin_idx, self._num_quantile_bins - 1))
                    new_hist[bin_idx] += count
            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        assert self._histograms is not None and self._bin_edges is not None
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self) -> list[np.ndarray]:
        assert self._histograms is not None and self._bin_edges is not None
        results = []
        for q in self._quantile_list:
            target_count = q * self._count
            q_values = [
                self._compute_single_quantile(hist, edges, target_count)
                for hist, edges in zip(self._histograms, self._bin_edges, strict=True)
            ]
            results.append(np.array(q_values))
        return results

    def _compute_single_quantile(self, hist: np.ndarray, edges: np.ndarray, target_count: float) -> float:
        cumsum = np.cumsum(hist)
        idx = int(np.searchsorted(cumsum, target_count))
        if idx == 0:
            return float(edges[0])
        if idx >= len(cumsum):
            return float(edges[-1])
        count_before = float(cumsum[idx - 1])
        count_in_bin = float(cumsum[idx] - count_before)
        if count_in_bin == 0.0:
            return float(edges[idx])
        fraction = (target_count - count_before) / count_in_bin
        return float(edges[idx] + fraction * (edges[idx + 1] - edges[idx]))


def compute_exact_vector_stats(arr: np.ndarray) -> dict[str, np.ndarray]:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
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


def protect_stationary_dims(
    stats: dict[str, np.ndarray],
    action_mask: np.ndarray,
    arm_min_range: float = DEFAULT_ARM_MIN_RANGE,
    arm_min_std: float = DEFAULT_ARM_MIN_STD,
) -> dict[str, np.ndarray]:
    """Widen near-constant dimensions so q01_q99 or mean_std normalization never blows up."""
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
            logging.info(
                "Widened dim %d (is_arm=%s) std %.5f -> %.4f",
                d,
                is_arm,
                float(out["std"][d]),
                min_std,
            )
            out["std"][d] = min_std
    return out


def aggregate_video_stats(
    per_ep_video_stats: list[dict[str, dict[str, list[Any]]]],
) -> dict[str, dict[str, list[Any]]]:
    """Aggregate per-episode video stats (count-weighted mean/var + min/max/quantiles)."""
    out: dict[str, dict[str, list[Any]]] = {}
    for key in CAMERA_TOPICS:
        items = [ep[key] for ep in per_ep_video_stats if key in ep]
        if not items:
            continue
        counts = np.array([int(x["count"][0]) for x in items], dtype=np.float64)
        total = float(counts.sum())
        weights = counts / max(total, 1.0)
        means = np.stack([np.asarray(x["mean"], dtype=np.float64) for x in items], axis=0)
        stds = np.stack([np.asarray(x["std"], dtype=np.float64) for x in items], axis=0)
        mins = np.stack([np.asarray(x["min"], dtype=np.float64) for x in items], axis=0)
        maxs = np.stack([np.asarray(x["max"], dtype=np.float64) for x in items], axis=0)

        w_shape = (-1,) + (1,) * (means.ndim - 1)
        w = weights.reshape(w_shape)
        global_mean = np.sum(w * means, axis=0)
        global_var = np.sum(w * (stds**2 + (means - global_mean) ** 2), axis=0)
        agg: dict[str, list[Any]] = {
            "min": np.min(mins, axis=0).tolist(),
            "max": np.max(maxs, axis=0).tolist(),
            "mean": global_mean.tolist(),
            "std": np.sqrt(np.maximum(global_var, 0.0)).tolist(),
            "count": [int(total)],
        }
        for qk in QUANTILE_KEYS:
            if all(qk in x for x in items):
                qs = np.stack([np.asarray(x[qk], dtype=np.float64) for x in items], axis=0)
                agg[qk] = np.sum(w * qs, axis=0).tolist()
        out[key] = agg
    return out


def compute_episode_rel_actions(
    states: np.ndarray,
    actions: np.ndarray,
    *,
    chunk_size: int,
    action_mask: np.ndarray,
    include_padded_tail: bool = False,
) -> np.ndarray:
    """Vectorized computation of chunk-relative actions for one episode.

    When `include_padded_tail=False` (default, matching `mask_action_pad: true`),
    only non-padded timesteps (`t + k < ep_len`) contribute to action statistics.
    """
    ep_len = states.shape[0]
    t_idx = np.arange(ep_len)[:, None]
    k_idx = np.arange(chunk_size)[None, :]
    step_idx = t_idx + k_idx
    clamped_idx = np.minimum(step_idx, ep_len - 1)
    ref = np.where(action_mask, states, 0.0)[:, None, :]
    rel_chunks = actions[clamped_idx] - ref
    if include_padded_tail:
        return rel_chunks.reshape(-1, actions.shape[1])
    valid_mask = step_idx < ep_len
    return rel_chunks[valid_mask]


def log_16dim_stats_summary(
    stats_json: dict[str, Any],
    joint_names: list[str],
    action_mask: np.ndarray,
) -> None:
    for feat_key in (OBS_STATE, ACTION):
        feat_stats = stats_json[feat_key]
        q01 = feat_stats["q01"]
        mean = feat_stats["mean"]
        q99 = feat_stats["q99"]
        std = feat_stats["std"]
        logging.info("[stats] === %s 16-dim summary ===", feat_key)
        for i, name in enumerate(joint_names):
            mode = "REL" if (feat_key == ACTION and bool(action_mask[i])) else "ABS"
            rng = float(q99[i]) - float(q01[i])
            logging.info(
                "  [%02d] %-19s (%s)  q01=%+8.4f  mean=%+8.4f  q99=%+8.4f  range=%8.4f  std=%7.4f",
                i,
                name,
                mode,
                q01[i],
                mean[i],
                q99[i],
                rng,
                std[i],
            )


def compute_final_global_stats(
    output_dir: Path,
    episode_rows: list[dict[str, Any]],
    per_ep_video_stats: list[dict[str, dict[str, list[Any]]]],
    *,
    chunk_size: int,
    arm_min_range: float,
    arm_min_std: float,
    include_padded_tail: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read episode parquets via chunk/file indices and compute float64 abs + VLASH relative stats."""
    all_states: list[np.ndarray] = []
    all_abs_actions: list[np.ndarray] = []
    scalar_cols: dict[str, list[np.ndarray]] = {
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "index": [],
        "task_index": [],
        "subtask_index": [],
    }

    total_rel_rows = 0
    for row in episode_rows:
        ep_len = int(row["length"])
        if include_padded_tail:
            total_rel_rows += ep_len * chunk_size
        else:
            # sum_{t=0}^{ep_len-1} min(chunk_size, ep_len - t)
            full_cnt = max(0, ep_len - chunk_size)
            tail_n = min(ep_len, chunk_size)
            total_rel_rows += full_cnt * chunk_size + (tail_n * (tail_n + 1)) // 2

    use_streaming_quantiles = total_rel_rows > EXACT_QUANTILE_ROW_LIMIT
    if use_streaming_quantiles:
        logging.info(
            "Relative action rows=%d > %d; using streaming float64 RunningQuantileStats.",
            total_rel_rows,
            EXACT_QUANTILE_ROW_LIMIT,
        )
    rel_runner = RunningQuantileStats() if use_streaming_quantiles else None
    all_rel_actions: list[np.ndarray] = []

    for row in episode_rows:
        chunk_idx = int(row["data/chunk_index"])
        file_idx = int(row["data/file_index"])
        pq_path = output_dir / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
        tbl = pq.read_table(pq_path)
        states = np.asarray(tbl[OBS_STATE].to_pylist(), dtype=np.float64)
        actions = np.asarray(tbl[ACTION].to_pylist(), dtype=np.float64)
        all_states.append(states)
        all_abs_actions.append(actions)
        for col in scalar_cols:
            scalar_cols[col].append(np.asarray(tbl[col].to_numpy(), dtype=np.float64))

        rel_ep = compute_episode_rel_actions(
            states,
            actions,
            chunk_size=chunk_size,
            action_mask=ACTION_MASK,
            include_padded_tail=include_padded_tail,
        )
        if rel_runner is not None:
            rel_runner.update(rel_ep)
        else:
            all_rel_actions.append(rel_ep)

    states_cat = np.concatenate(all_states, axis=0)
    abs_actions_cat = np.concatenate(all_abs_actions, axis=0)

    state_stats = protect_stationary_dims(
        compute_exact_vector_stats(states_cat),
        action_mask=ACTION_MASK,
        arm_min_range=arm_min_range,
        arm_min_std=arm_min_std,
    )
    abs_action_stats = protect_stationary_dims(
        compute_exact_vector_stats(abs_actions_cat),
        action_mask=ACTION_MASK,
        arm_min_range=arm_min_range,
        arm_min_std=arm_min_std,
    )
    raw_rel_stats = (
        rel_runner.get_statistics()
        if rel_runner is not None
        else compute_exact_vector_stats(np.concatenate(all_rel_actions, axis=0))
    )
    rel_action_stats = protect_stationary_dims(
        raw_rel_stats,
        action_mask=ACTION_MASK,
        arm_min_range=arm_min_range,
        arm_min_std=arm_min_std,
    )

    base_stats: dict[str, Any] = aggregate_video_stats(per_ep_video_stats)
    for col, parts in scalar_cols.items():
        cat = np.concatenate(parts, axis=0)
        base_stats[col] = {k: v.tolist() for k, v in compute_exact_vector_stats(cat).items()}
    base_stats[OBS_STATE] = {k: v.tolist() for k, v in state_stats.items()}

    abs_stats = dict(base_stats)
    abs_stats[ACTION] = {k: v.tolist() for k, v in abs_action_stats.items()}

    rel_stats = dict(base_stats)
    rel_stats[ACTION] = {k: v.tolist() for k, v in rel_action_stats.items()}
    return rel_stats, abs_stats


# ============================================================================
# 8. Source Discovery, Exclusion & Config Fingerprint Progress
# ============================================================================
def _is_hidden_path(path: Path, relative_to: Path) -> bool:
    for part in path.relative_to(relative_to).parts:
        if part.startswith(".") or part.startswith("._"):
            return True
    return False


def discover_episodes(
    source_root: Path,
    *,
    task_mode: str,
    unified_task: str,
    task_map: dict[str, str],
    default_task: str,
    sync_only: bool = False,
) -> list[tuple[str, Path, str]]:
    """Discover *.mcap episodes across task folders (supports flat and nested layouts)."""
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root directory not found: {source_root}")

    episodes: list[tuple[str, Path, str]] = []
    subdirs = [d for d in sorted(source_root.iterdir()) if d.is_dir() and not d.name.startswith(".")]
    if not subdirs:
        subdirs = [source_root]

    for folder in subdirs:
        task_name = unified_task if task_mode == "unified" else task_map.get(folder.name, default_task)

        all_mcaps = [
            p
            for p in sorted(folder.rglob("*.mcap"))
            if p.is_file() and not _is_hidden_path(p, folder)
        ]
        if sync_only:
            selected = [p for p in all_mcaps if "SYNC" in p.name]
        else:
            chosen_by_key: dict[tuple[Path, str], Path] = {}
            for p in all_mcaps:
                if p.name.endswith(".FIN.mcap"):
                    stem = p.name[: -len(".FIN.mcap")]
                    chosen_by_key[(p.parent, stem)] = p
            for p in all_mcaps:
                if p.name.endswith(".SYNC.mcap"):
                    stem = p.name[: -len(".SYNC.mcap")]
                    key = (p.parent, stem)
                    if key not in chosen_by_key:
                        chosen_by_key[key] = p
            if not chosen_by_key:
                for p in all_mcaps:
                    chosen_by_key[(p.parent, p.stem)] = p
            selected = sorted(chosen_by_key.values(), key=lambda p: p.relative_to(folder).as_posix())

        for p in selected:
            episodes.append((task_name, p, folder.name))
    return episodes


def load_exclude_patterns(
    exclude_files: list[str] | None,
    exclude_file_list: Path | None,
) -> list[str]:
    patterns: list[str] = []
    for raw in exclude_files or []:
        for tok in raw.split(","):
            tok = tok.strip()
            if tok:
                patterns.append(tok)
    if exclude_file_list is not None:
        if not exclude_file_list.is_file():
            raise FileNotFoundError(f"Exclude file list not found: {exclude_file_list}")
        for line in exclude_file_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return list(dict.fromkeys(patterns))


def matches_exclude_pattern(mcap_path: Path, rel_id: str, patterns: list[str]) -> str | None:
    if not patterns:
        return None
    name_l = mcap_path.name.lower()
    rel_l = rel_id.lower()
    for pat in patterns:
        pat_l = pat.lower()
        if pat_l == name_l or pat_l in name_l or pat_l in rel_l:
            return pat
    return None


def compute_config_fingerprint(
    args: argparse.Namespace,
    task_map: dict[str, str],
    exclude_patterns: list[str],
) -> str:
    payload = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "joint_groups": JOINT_GROUPS,
        "camera_keys": sorted(CAMERA_TOPICS.keys()),
        "task_mode": args.task_mode,
        "unified_task": args.unified_task,
        "default_task": args.default_task,
        "task_map": task_map if args.task_mode == "per_folder" else {},
        "sync_only": bool(args.sync_only),
        "exclude_patterns": sorted(exclude_patterns),
        "fps": int(args.fps),
        "resample_uniform": bool(args.resample_uniform),
        "target_resolution": str(args.target_resolution),
        "resize_mode": str(args.resize_mode),
        "codec": str(args.codec),
        "crf": int(args.crf),
        "gop": int(args.gop),
        "preset": str(args.preset),
        "min_frames": int(args.min_frames),
        "arm_tolerance_ms": float(args.arm_tolerance_ms),
        "arm_startup_grace_s": float(args.arm_startup_grace_s),
        "arm_startup_max_gap_ms": float(args.arm_startup_max_gap_ms),
        "hold_eps_rad": float(args.hold_eps_rad),
        "allow_missing_groups": bool(args.allow_missing_groups),
        "trim_leading_idle": bool(args.trim_leading_idle),
        "split_on_interior_gaps": bool(args.split_on_interior_gaps),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def new_progress(fingerprint: str, fps_override: int | None) -> dict[str, Any]:
    return {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "config_fingerprint": fingerprint,
        "processed_identifiers": [],
        "failed_identifiers": [],
        "excluded_files": [],
        "episode_rows": [],
        "per_ep_video_stats": [],
        "video_verify_entries": [],
        "total_frames": 0,
        "total_episodes": 0,
        "first_video_infos": None,
        "first_height": 224,
        "first_width": 224,
        "fps": fps_override if fps_override is not None else None,
        "action_dim": 16,
        "state_dim": 16,
        "conversion_entries": [],
        "tasks": [],
        "subtasks": ["complete task"],
        "letterbox_infos": {},
        "raw_camera_shapes": {},
        "stages": {
            "finalized": False,
            "verified": False,
        },
    }


def load_progress(output_dir: Path, fingerprint: str, fps_override: int | None) -> dict[str, Any]:
    progress_path = output_dir / "progress.json"
    if not progress_path.exists():
        return new_progress(fingerprint, fps_override)
    with progress_path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    if state.get("schema_version") != PROGRESS_SCHEMA_VERSION:
        raise ValueError(
            f"Incompatible progress.json schema_version={state.get('schema_version')}; "
            f"use a different --output-dir or pass --force."
        )
    if state.get("config_fingerprint") != fingerprint:
        raise ValueError(
            "Config mismatch against existing progress.json (refusing to mix configurations).\n"
            f"  existing: {state.get('config_fingerprint')}\n"
            f"  current : {fingerprint}\n"
            "  Pass --force with --step all to rebuild cleanly, or use a new --output-dir."
        )
    return state


def save_progress(output_dir: Path, progress: dict[str, Any]) -> None:
    write_json(output_dir / "progress.json", progress)


# ============================================================================
# 9. Pipeline Stages: convert / finalize / verify
# ============================================================================
def run_convert(
    args: argparse.Namespace,
    task_map: dict[str, str],
    exclude_patterns: list[str],
    progress: dict[str, Any],
) -> None:
    all_eps = discover_episodes(
        args.source_root,
        task_mode=args.task_mode,
        unified_task=args.unified_task,
        task_map=task_map,
        default_task=args.default_task,
        sync_only=args.sync_only,
    )
    if not all_eps:
        raise FileNotFoundError(f"No .mcap files found under {args.source_root}")

    processed_ids = set(progress["processed_identifiers"])
    failed_ids = set(progress["failed_identifiers"])
    if args.retry_failed and failed_ids:
        logging.info("[convert] --retry-failed: clearing %d previous failures", len(failed_ids))
        failed_ids.clear()

    tasks: list[str] = list(progress["tasks"])
    task_index_map = {t: i for i, t in enumerate(tasks)}
    subtasks: list[str] = list(progress["subtasks"])
    subtask_index_map = {s: i for i, s in enumerate(subtasks)}

    episode_rows: list[dict[str, Any]] = list(progress["episode_rows"])
    per_ep_video_stats: list[dict[str, Any]] = list(progress["per_ep_video_stats"])
    video_verify_entries: list[dict[str, Any]] = list(progress["video_verify_entries"])
    conversion_entries: list[dict[str, Any]] = list(progress["conversion_entries"])
    total_frames = int(progress["total_frames"])
    total_episodes = int(progress["total_episodes"])
    first_video_infos = (
        {k: VideoInfo(**v) for k, v in progress["first_video_infos"].items()}
        if progress["first_video_infos"]
        else None
    )
    first_height = int(progress["first_height"])
    first_width = int(progress["first_width"])
    locked_fps: int | None = int(progress["fps"]) if progress["fps"] is not None else None
    action_dim = int(progress["action_dim"])
    state_dim = int(progress["state_dim"])
    letterbox_infos = dict(progress.get("letterbox_infos", {}))
    raw_camera_shapes = dict(progress.get("raw_camera_shapes", {}))

    target_size = parse_target_resolution(args.target_resolution)
    excluded_records: list[dict[str, str]] = []
    folder_kept_counts: dict[str, int] = defaultdict(int)
    done_per_folder: dict[str, int] = defaultdict(int)
    for ident in processed_ids:
        folder_key = ident.split("/", 1)[0]
        done_per_folder[folder_key] += 1

    jobs: list[dict[str, Any]] = []
    staging_dir = args.output_dir / "_staging_videos"
    staging_dir.mkdir(parents=True, exist_ok=True)

    for task_name, mcap_path, folder_name in all_eps:
        rel_path_str = mcap_path.relative_to(args.source_root).as_posix()
        ident = f"{folder_name}/{mcap_path.stem}"
        matched = matches_exclude_pattern(mcap_path, rel_path_str, exclude_patterns)
        if matched is not None:
            excluded_records.append(
                {"folder": folder_name, "file": rel_path_str, "matched_pattern": matched}
            )
            continue
        folder_kept_counts[folder_name] += 1
        if ident in processed_ids or ident in failed_ids:
            continue
        if args.max_episodes_per_folder is not None and done_per_folder[folder_name] >= args.max_episodes_per_folder:
            continue
        done_per_folder[folder_name] += 1
        jobs.append(
            {
                "ident": ident,
                "mcap_path": str(mcap_path),
                "task_name": task_name,
                "staging_dir": str(staging_dir),
                "fps_override": locked_fps,
                "min_frames": args.min_frames,
                "target_size": list(target_size) if target_size is not None else None,
                "resize_mode": args.resize_mode,
                "resample_uniform": args.resample_uniform,
                "codec": args.codec,
                "crf": args.crf,
                "gop": args.gop,
                "preset": args.preset,
                "no_video_verify": args.no_video_verify,
                "max_first_frame_mae": args.max_first_frame_mae,
                "max_pad_mean": args.max_pad_mean,
                "pts_tolerance_s": args.pts_tolerance_s,
                "skip_videos": args.skip_videos,
                "arm_tolerance_ms": args.arm_tolerance_ms,
                "arm_startup_grace_s": args.arm_startup_grace_s,
                "arm_startup_max_gap_ms": args.arm_startup_max_gap_ms,
                "hold_eps_rad": args.hold_eps_rad,
                "allow_missing_groups": args.allow_missing_groups,
                "trim_leading_idle": args.trim_leading_idle,
                "split_on_interior_gaps": args.split_on_interior_gaps,
            }
        )

    progress["excluded_files"] = excluded_records
    logging.info(
        "Discovered %d MCAPs across folders %s (excluded=%d, pending=%d, task_mode=%s)",
        len(all_eps),
        dict(folder_kept_counts),
        len(excluded_records),
        len(jobs),
        args.task_mode,
    )

    if not jobs:
        shutil.rmtree(staging_dir, ignore_errors=True)
        logging.info("[convert] Nothing pending (done=%d, failed=%d).", len(processed_ids), len(failed_ids))
        return

    def _consume_result(res: dict[str, Any]) -> None:
        nonlocal total_frames, total_episodes, first_video_infos
        nonlocal first_height, first_width, locked_fps, action_dim, state_dim

        ident = res["ident"]
        if not res.get("ok"):
            logging.warning("Skipping %s: %s", ident, res.get("error"))
            failed_ids.add(ident)
        else:
            snap_episodes = total_episodes
            snap_frames = total_frames
            snap_rows = len(episode_rows)
            snap_vstats = len(per_ep_video_stats)
            snap_vmetrics = len(video_verify_entries)
            snap_conv = len(conversion_entries)
            created_files: list[Path] = []
            task_name = res["task_name"]

            try:
                if task_name not in task_index_map:
                    task_index_map[task_name] = len(tasks)
                    tasks.append(task_name)
                task_idx = task_index_map[task_name]

                segments = res["segments"]
                for seg in segments:
                    ep_index = total_episodes
                    chunk_idx, file_idx = chunk_and_file_index(ep_index, CHUNKS_SIZE)
                    state = seg["state"]
                    action = seg["action"]
                    timestamp = seg["timestamp"]
                    frame_index = seg["frame_index"]
                    num_frames = int(seg["num_frames"])
                    ep_fps = int(seg["fps"])
                    if locked_fps is None:
                        locked_fps = ep_fps
                    elif ep_fps != locked_fps:
                        raise ValueError(f"Inconsistent episode FPS {ep_fps} != dataset FPS {locked_fps}")
                    state_dim = int(state.shape[1])
                    action_dim = int(action.shape[1])

                    v_infos = {k: VideoInfo(**v) for k, v in seg["video_infos"].items()}
                    if first_video_infos is None and v_infos:
                        first_video_infos = v_infos
                        first_height = next(iter(v_infos.values())).height
                        first_width = next(iter(v_infos.values())).width
                    letterbox_infos.update(seg["letterbox_infos"])
                    raw_camera_shapes.update(seg["camera_shapes"])

                    for video_key, tmp_path_str in seg["tmp_videos"].items():
                        final_dst = (
                            args.output_dir
                            / "videos"
                            / video_key
                            / f"chunk-{chunk_idx:03d}"
                            / f"file-{file_idx:03d}.mp4"
                        )
                        final_dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(tmp_path_str, final_dst)
                        created_files.append(final_dst)

                    subtask_index = np.full(num_frames, subtask_index_map["complete task"], dtype=np.int64)
                    ep_idx_arr = np.full(num_frames, ep_index, dtype=np.int64)
                    global_index = np.arange(total_frames, total_frames + num_frames, dtype=np.int64)
                    task_idx_arr = np.full(num_frames, task_idx, dtype=np.int64)

                    data_path = (
                        args.output_dir / "data" / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.parquet"
                    )
                    write_episode_parquet(
                        data_path,
                        state=state,
                        action=action,
                        subtask_index=subtask_index,
                        timestamp=timestamp,
                        frame_index=frame_index,
                        episode_index=ep_idx_arr,
                        index=global_index,
                        task_index=task_idx_arr,
                    )
                    created_files.append(data_path)

                    per_ep_video_stats.append(seg["video_stats"])
                    for vm in seg["verify_metrics"]:
                        video_verify_entries.append({"episode_index": ep_index, **vm})

                    row: dict[str, Any] = {
                        "episode_index": ep_index,
                        "tasks": [task_name],
                        "length": num_frames,
                        "data/chunk_index": chunk_idx,
                        "data/file_index": file_idx,
                        "dataset_from_index": total_frames,
                        "dataset_to_index": total_frames + num_frames,
                        "meta/episodes/chunk_index": 0,
                        "meta/episodes/file_index": 0,
                    }
                    for video_key in CAMERA_TOPICS:
                        row[f"videos/{video_key}/chunk_index"] = chunk_idx
                        row[f"videos/{video_key}/file_index"] = file_idx
                        row[f"videos/{video_key}/from_timestamp"] = 0.0
                        row[f"videos/{video_key}/to_timestamp"] = num_frames / float(locked_fps)
                    episode_rows.append(row)

                    src_ep_id = ident if len(segments) == 1 else f"{ident}#seg{seg['seg_idx']}"
                    conversion_entries.append(
                        {
                            "episode_index": ep_index,
                            "source_episode_id": src_ep_id,
                            "frames": num_frames,
                            "task": task_name,
                            "timing": seg["timing"],
                            "alignment": seg["alignment"],
                        }
                    )
                    total_frames += num_frames
                    total_episodes += 1

                processed_ids.add(ident)
                progress["stages"]["finalized"] = False
                progress["stages"]["verified"] = False
            except Exception as exc:
                logging.warning("Rollback %s after consume error: %s", ident, exc)
                total_episodes = snap_episodes
                total_frames = snap_frames
                del episode_rows[snap_rows:]
                del per_ep_video_stats[snap_vstats:]
                del video_verify_entries[snap_vmetrics:]
                del conversion_entries[snap_conv:]
                for orphan in created_files:
                    orphan.unlink(missing_ok=True)
                failed_ids.add(ident)

        progress.update(
            {
                "processed_identifiers": sorted(processed_ids),
                "failed_identifiers": sorted(failed_ids),
                "episode_rows": episode_rows,
                "per_ep_video_stats": per_ep_video_stats,
                "video_verify_entries": video_verify_entries,
                "total_frames": total_frames,
                "total_episodes": total_episodes,
                "first_video_infos": (
                    {k: asdict(v) for k, v in first_video_infos.items()}
                    if first_video_infos
                    else None
                ),
                "first_height": first_height,
                "first_width": first_width,
                "fps": locked_fps,
                "action_dim": action_dim,
                "state_dim": state_dim,
                "conversion_entries": conversion_entries,
                "tasks": tasks,
                "subtasks": subtasks,
                "letterbox_infos": letterbox_infos,
                "raw_camera_shapes": raw_camera_shapes,
            }
        )
        save_progress(args.output_dir, progress)

    if locked_fps is None and args.workers > 1:
        # Lock auto-detected FPS from first job before fanning out workers
        first_job = jobs.pop(0)
        _consume_result(_convert_episode_worker(first_job))
        for j in jobs:
            j["fps_override"] = locked_fps

    if jobs:
        if args.workers <= 1:
            it = tqdm(jobs, desc="Converting MCAPs") if tqdm else jobs
            for job in it:
                job["fps_override"] = locked_fps
                _consume_result(_convert_episode_worker(job))
        else:
            ctx = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers, mp_context=ctx
            ) as executor:
                results_iter = executor.map(_convert_episode_worker, jobs)
                if tqdm:
                    results_iter = tqdm(
                        results_iter, total=len(jobs), desc=f"Converting MCAPs ({args.workers}w)"
                    )
                for res in results_iter:
                    _consume_result(res)

    shutil.rmtree(staging_dir, ignore_errors=True)
    if total_episodes == 0:
        raise RuntimeError("Zero valid episodes converted.")


def run_finalize(args: argparse.Namespace, progress: dict[str, Any]) -> None:
    if not args.force and progress["stages"].get("finalized"):
        logging.info("[finalize] Already finalized; skipping (pass --force to recompute).")
        return

    total_episodes = int(progress["total_episodes"])
    total_frames = int(progress["total_frames"])
    if total_episodes == 0:
        raise RuntimeError("[finalize] Zero valid episodes in progress.json; run convert first.")

    episode_rows = progress["episode_rows"]
    per_ep_video_stats = progress["per_ep_video_stats"]
    tasks: list[str] = list(progress["tasks"])
    subtasks: list[str] = list(progress["subtasks"])
    resolved_fps = int(progress["fps"] or DEFAULT_FPS)
    first_height = int(progress["first_height"] or 224)
    first_width = int(progress["first_width"] or 224)
    state_dim = int(progress["state_dim"] or 16)
    action_dim = int(progress["action_dim"] or 16)
    first_video_infos = (
        {k: VideoInfo(**v) for k, v in progress["first_video_infos"].items()}
        if progress["first_video_infos"]
        else None
    )

    logging.info(
        "[finalize] Computing float64 global & VLASH relative stats (chunk_size=%d, include_padded_tail=%s)...",
        args.chunk_size,
        args.include_padded_tail_in_stats,
    )
    rel_stats, abs_stats = compute_final_global_stats(
        args.output_dir,
        episode_rows,
        per_ep_video_stats,
        chunk_size=args.chunk_size,
        arm_min_range=args.arm_min_range,
        arm_min_std=args.arm_min_std,
        include_padded_tail=args.include_padded_tail_in_stats,
    )
    log_16dim_stats_summary(rel_stats, ARM_STATE_ACTION_NAMES, ACTION_MASK)

    meta_dir = args.output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    features = make_lerobot_features(
        height=first_height,
        width=first_width,
        action_dim=action_dim,
        state_dim=state_dim,
        video_infos=first_video_infos,
        fps=resolved_fps,
    )
    info_payload = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": args.robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_subtasks": len(subtasks),
        "chunks_size": CHUNKS_SIZE,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": resolved_fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
        "repo_id": args.repo_id,
    }
    write_json(meta_dir / "info.json", info_payload)
    write_json(meta_dir / "stats.json", rel_stats)
    write_json(meta_dir / "stats.json.abs_bak", abs_stats)

    tasks_df = pd.DataFrame({"task_index": list(range(len(tasks)))}, index=tasks)
    tasks_df.index.name = "task"
    tasks_df.to_parquet(meta_dir / "tasks.parquet")

    subtasks_df = pd.DataFrame({"subtask_index": list(range(len(subtasks)))}, index=subtasks)
    subtasks_df.index.name = "subtask"
    subtasks_df.to_parquet(meta_dir / "subtasks.parquet")

    episodes_path = meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(episode_rows).to_parquet(episodes_path, index=False)

    verify_entries = progress.get("video_verify_entries", [])
    max_mae = max(
        (e["first_frame_mae"] for e in verify_entries if not math.isnan(e["first_frame_mae"])),
        default=float("nan"),
    )
    max_pad = max(
        (e["pad_mean"] for e in verify_entries if not math.isnan(e["pad_mean"])),
        default=float("nan"),
    )
    max_pts = max(
        (e["max_pts_err_s"] for e in verify_entries if not math.isnan(e["max_pts_err_s"])),
        default=float("nan"),
    )
    write_json(
        args.output_dir / "conversion_report.json",
        {
            "pipeline": "convert_mcap_to_internvla_lerobot_v3.py",
            "source_root": str(args.source_root),
            "output_dir": str(args.output_dir),
            "repo_id": args.repo_id,
            "robot_type": args.robot_type,
            "fps": resolved_fps,
            "chunk_size": args.chunk_size,
            "include_padded_tail_in_stats": bool(args.include_padded_tail_in_stats),
            "task_mode": args.task_mode,
            "tasks": tasks,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "excluded_files": progress.get("excluded_files", []),
            "failed_identifiers": progress["failed_identifiers"],
            "raw_camera_shapes": progress.get("raw_camera_shapes", {}),
            "letterbox_infos": progress.get("letterbox_infos", {}),
            "video_encoding": {
                "codec": args.codec,
                "crf": args.crf,
                "gop": args.gop,
                "preset": args.preset,
                "resize_mode": args.resize_mode,
                "max_first_frame_mae": None if math.isnan(max_mae) else round(max_mae, 4),
                "max_padding_mean": None if math.isnan(max_pad) else round(max_pad, 4),
                "max_pts_err_s": None if math.isnan(max_pts) else max_pts,
            },
            "episodes": progress["conversion_entries"],
        },
    )

    progress["stages"]["finalized"] = True
    progress["stages"]["verified"] = False
    save_progress(args.output_dir, progress)
    logging.info(
        "[finalize] Metadata & stats written: %d tasks, %d episodes, %d frames -> %s",
        len(tasks),
        total_episodes,
        total_frames,
        args.output_dir,
    )


def run_verify(args: argparse.Namespace, progress: dict[str, Any]) -> None:
    if not progress["stages"].get("finalized"):
        raise RuntimeError("[verify] Dataset is not finalized yet.")

    info = json.loads((args.output_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    stats = json.loads((args.output_dir / "meta" / "stats.json").read_text(encoding="utf-8"))
    abs_stats = json.loads((args.output_dir / "meta" / "stats.json.abs_bak").read_text(encoding="utf-8"))

    for key in (OBS_STATE, ACTION):
        feat = info["features"][key]
        if list(feat["shape"]) != [16]:
            raise ValueError(f"info.json {key}.shape={feat['shape']} != [16]")
        if list(feat["names"]) != ARM_STATE_ACTION_NAMES:
            raise ValueError(f"info.json {key}.names mismatch: {feat['names']}")

    target_size = parse_target_resolution(args.target_resolution)
    for cam_key in CAMERA_TOPICS:
        if cam_key not in info["features"]:
            raise ValueError(f"Missing camera feature {cam_key} in info.json")
        if target_size is not None:
            exp_shape = [target_size[1], target_size[0], 3]
            if list(info["features"][cam_key]["shape"]) != exp_shape:
                raise ValueError(f"{cam_key}.shape={info['features'][cam_key]['shape']} != {exp_shape}")

    tasks_df = pd.read_parquet(args.output_dir / "meta" / "tasks.parquet")
    if set(tasks_df.index.astype(str).tolist()) != set(progress["tasks"]):
        raise ValueError("tasks.parquet index does not match progress tasks")

    for key in (OBS_STATE, ACTION):
        for stat_name in ("min", "max", "mean", "std", "q01", "q99"):
            arr = np.asarray(stats[key][stat_name], dtype=np.float64)
            if arr.shape != (16,) or not np.all(np.isfinite(arr)):
                raise ValueError(f"Invalid stats.json[{key}][{stat_name}]")
        q01 = np.asarray(stats[key]["q01"], dtype=np.float64)
        q99 = np.asarray(stats[key]["q99"], dtype=np.float64)
        std = np.asarray(stats[key]["std"], dtype=np.float64)
        if np.any((q99 - q01) < 0.05) or np.any(std < 0.01):
            raise ValueError(f"Degenerate quantile/std in stats.json[{key}]")

    # Verify grippers [7, 15] stayed absolute in stats.json (min/max match abs_stats; chunk weighting can shift mean slightly)
    for g_idx in (7, 15):
        if (
            abs(stats[ACTION]["min"][g_idx] - abs_stats[ACTION]["min"][g_idx]) > 1e-4
            or abs(stats[ACTION]["max"][g_idx] - abs_stats[ACTION]["max"][g_idx]) > 1e-4
        ):
            raise ValueError(f"Gripper dim {g_idx} min/max altered (expected absolute range)!")

    ep_rows = progress["episode_rows"]
    row_by_ep = {int(r["episode_index"]): r for r in ep_rows}
    entries_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in progress["conversion_entries"]:
        entries_by_task[entry["task"]].append(entry)

    checked_videos = 0
    max_pad_seen = 0.0
    max_pts_seen = 0.0
    resolved_fps = int(info["fps"])

    for _, entries in sorted(entries_by_task.items()):
        entries_sorted = sorted(entries, key=lambda e: e["episode_index"])
        if args.verify_samples <= 0:
            picks = entries_sorted
        else:
            mid = len(entries_sorted) // 2
            picks = entries_sorted[mid : mid + args.verify_samples]

        for entry in picks:
            ep_idx = int(entry["episode_index"])
            row = row_by_ep[ep_idx]
            c_idx = int(row["data/chunk_index"])
            f_idx = int(row["data/file_index"])
            pq_path = args.output_dir / "data" / f"chunk-{c_idx:03d}" / f"file-{f_idx:03d}.parquet"
            tbl = pq.read_table(pq_path)
            if tbl.num_rows != int(entry["frames"]):
                raise ValueError(f"{pq_path}: rows={tbl.num_rows} != {entry['frames']}")

            if not args.skip_videos:
                raw_shapes = progress.get("raw_camera_shapes") or {}
                for cam_key in CAMERA_TOPICS:
                    vc = int(row[f"videos/{cam_key}/chunk_index"])
                    vf = int(row[f"videos/{cam_key}/file_index"])
                    vid_path = (
                        args.output_dir / "videos" / cam_key / f"chunk-{vc:03d}" / f"file-{vf:03d}.mp4"
                    )
                    v_info, first_rgb, pts_err_s, _ = decode_and_verify_video_pyav(
                        vid_path, expected_fps=float(resolved_fps)
                    )
                    if v_info.frames != int(entry["frames"]):
                        raise ValueError(f"{vid_path}: frames={v_info.frames} != {entry['frames']}")
                    if pts_err_s >= args.pts_tolerance_s:
                        raise ValueError(f"{vid_path}: pts_err={pts_err_s} >= {args.pts_tolerance_s}")
                    max_pts_seen = max(max_pts_seen, pts_err_s)
                    if target_size is not None:
                        if (v_info.width, v_info.height) != target_size:
                            raise ValueError(f"{vid_path}: size {v_info.width}x{v_info.height} != {target_size}")
                        if args.resize_mode == "pad" and cam_key in raw_shapes:
                            src_h, src_w = raw_shapes[cam_key]
                            regions = _padding_slices(src_w, src_h, target_size[0], target_size[1])
                            if regions:
                                pad_pixels = np.concatenate(
                                    [first_rgb[r, c].reshape(-1, 3) for r, c in regions], axis=0
                                )
                                pad_mean = float(pad_pixels.mean())
                                max_pad_seen = max(max_pad_seen, pad_mean)
                                if pad_mean > args.max_pad_mean:
                                    raise ValueError(f"{vid_path}: pad_mean={pad_mean:.3f} > {args.max_pad_mean}")
                    checked_videos += 1

    if not args.skip_videos:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            logging.info("[verify] lerobot not installed locally; skipping LeRobotDataset load check.")
        else:
            ds = LeRobotDataset(repo_id=info["repo_id"], root=args.output_dir, video_backend="pyav")
            if ds.meta.total_episodes != int(info["total_episodes"]) or ds.meta.total_frames != int(info["total_frames"]):
                raise ValueError("LeRobotDataset metadata mismatch")
            sample = ds[0]
            assert tuple(sample[OBS_STATE].shape) == (16,)
            assert tuple(sample[ACTION].shape) == (16,)
            for cam_key in CAMERA_TOPICS:
                expected_hw = (int(info["features"][cam_key]["shape"][0]), int(info["features"][cam_key]["shape"][1]))
                if tuple(sample[cam_key].shape) != (3, *expected_hw):
                    raise RuntimeError(f"Camera {cam_key} shape mismatch: {tuple(sample[cam_key].shape)}")
                if float(sample[cam_key].max()) <= 0.0:
                    raise RuntimeError(f"Camera {cam_key} decoded as all zeros (silent video decode fallback)")
            logging.info("[verify] LeRobotDataset(pyav) sample load PASSED!")

    progress["stages"]["verified"] = True
    save_progress(args.output_dir, progress)

    logging.info(
        "[verify] PASSED! episodes=%d frames=%d tasks=%d checked_videos=%d max_pad_mean=%.3f max_pts_err=%.2e",
        info["total_episodes"],
        info["total_frames"],
        info["total_tasks"],
        checked_videos,
        max_pad_seen,
        max_pts_seen,
    )


# ============================================================================
# 10. Built-in End-to-End Self-Test (--self-test)
# ============================================================================
def run_self_test() -> None:
    logging.info("[self-test] Running synthetic end-to-end verification...")
    assert chunk_and_file_index(1005, CHUNKS_SIZE) == (1, 5)
    assert len(ARM_STATE_ACTION_NAMES) == 16
    assert ARM_STATE_ACTION_NAMES[0] == "left_arm_joint1"
    assert ARM_STATE_ACTION_NAMES[7] == "left_arm_gripper"
    assert ARM_STATE_ACTION_NAMES[8] == "right_arm_joint1"
    assert ARM_STATE_ACTION_NAMES[15] == "right_arm_gripper"

    with tempfile.TemporaryDirectory(prefix="internvla_parcel_selftest_") as tmp:
        root = Path(tmp)
        output_dir = root / "out"

        # 1. Verify letterbox + libx264 crf=18 + PTS + padding + stats
        frames_bgr = []
        for idx in range(70):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img[:, :, 0] = (idx * 3) % 255
            img[:, :, 1] = 120
            img[:, :, 2] = 200
            frames_bgr.append(img)

        vid_infos: dict[str, VideoInfo] = {}
        ep_vid_stats: dict[str, dict[str, list[Any]]] = {}
        for cam_key in CAMERA_TOPICS:
            dst = output_dir / "videos" / cam_key / "chunk-000" / "file-000.mp4"
            v_info, v_stats, v_metrics = encode_bgr_frames_h264(
                frames_bgr,
                dst,
                camera_key=cam_key,
                fps=30,
                target_size=(224, 224),
                resize_mode="pad",
            )
            assert (v_info.width, v_info.height) == (224, 224)
            assert v_info.frames == 70
            assert v_metrics.max_pts_err_s < DEFAULT_PTS_TOLERANCE_S
            assert v_metrics.pad_mean <= DEFAULT_MAX_PAD_MEAN
            vid_infos[cam_key] = v_info
            ep_vid_stats[cam_key] = {sk: np.asarray(sv).tolist() for sk, sv in v_stats.items()}

        # 2. Verify Causal ZOH: sequential bimanual handoff + tail hold rescue + gap split
        fps = 30
        n_seq = 220
        master_ts = np.arange(n_seq, dtype=np.int64) * int(1e9 // fps)
        sensor_seq = np.zeros((n_seq, 16), dtype=np.float32)
        sensor_seq[45:, 0] = np.linspace(0.0, 1.2, n_seq - 45, dtype=np.float32)   # left_arm moves at 45
        sensor_seq[130:, 8] = np.linspace(0.0, 0.8, n_seq - 130, dtype=np.float32)  # right_arm moves at 130
        r_idx = np.r_[np.array([130, 131]), np.arange(145, n_seq)]
        r_tgts = []
        for i in r_idx:
            pos = sensor_seq[i, 8:15].copy()
            if i in (130, 131):
                pos[0] = sensor_seq[145, 8]
            r_tgts.append((int(master_ts[i]), pos))
        seq_targets = {
            "left_arm": [(int(master_ts[i]), sensor_seq[i, :7].copy()) for i in range(45, n_seq)],
            "left_gripper": [(int(master_ts[45]), np.array([100.0], dtype=np.float32))],
            "right_arm": r_tgts,
            "right_gripper": [(int(master_ts[130]), np.array([100.0], dtype=np.float32))],
        }
        seq_runs, seq_cidx, _, _ = validate_and_segment_action_alignment(
            master_ts, master_ts, sensor_seq, seq_targets, JOINT_GROUPS, episode_name="seq.mcap"
        )
        assert len(seq_runs) == 1 and seq_runs[0].start == 45 and seq_runs[0].stop == n_seq
        assert np.all(seq_cidx["right_arm"][45:130] == -1)

        # Tail hold rescue (like 20261126_140543): right_arm target stops 1.9s early with tiny drift <= 0.05
        n_hold = 180
        ts_hold = np.arange(n_hold, dtype=np.int64) * int(1e9 // fps)
        s_hold = np.zeros((n_hold, 16), dtype=np.float32)
        s_hold[:, :7] = np.linspace(0.0, 1.0, n_hold, dtype=np.float32)[:, None]
        s_hold[:120, 8:15] = np.linspace(0.0, 0.5, 120, dtype=np.float32)[:, None]
        s_hold[120:, 8:15] = 0.505  # 0.005 rad drift <= 0.05
        hold_targets = {
            "left_arm": [(int(ts_hold[i]), s_hold[i, :7].copy()) for i in range(n_hold)],
            "left_gripper": [(int(ts_hold[0]), np.array([100.0], dtype=np.float32))],
            "right_arm": [(int(ts_hold[i]), s_hold[i, 8:15].copy()) for i in range(120)],
            "right_gripper": [(int(ts_hold[0]), np.array([100.0], dtype=np.float32))],
        }
        hold_runs, _, _, hold_rep = validate_and_segment_action_alignment(
            ts_hold, ts_hold, s_hold, hold_targets, JOINT_GROUPS, episode_name="hold.mcap"
        )
        assert len(hold_runs) == 1 and hold_runs[0] == slice(0, n_hold)
        assert hold_rep["action_groups"]["right_arm"]["hold_rescued_frames"] > 0

        # Mid-trajectory gap split + missing arm fill
        n_gap = 200
        ts_gap = np.arange(n_gap, dtype=np.int64) * int(1e9 // fps)
        s_gap = np.zeros((n_gap, 16), dtype=np.float32)
        s_gap[:, 0] = np.linspace(0.0, 2.0, n_gap, dtype=np.float32)
        keep_t = np.r_[np.arange(0, 80), np.arange(120, n_gap)]
        gap_targets = {
            "left_arm": [(int(ts_gap[i]), s_gap[i, :7].copy()) for i in keep_t],
            "left_gripper": [(int(ts_gap[0]), np.array([100.0], dtype=np.float32))],
            "right_arm": [],
            "right_gripper": [],
        }
        gap_runs, _, _, gap_rep = validate_and_segment_action_alignment(
            ts_gap, ts_gap, s_gap, gap_targets, JOINT_GROUPS, episode_name="gap.mcap"
        )
        assert len(gap_runs) == 2
        assert "right_arm" in gap_rep["missing_groups_filled_from_state"]

        # 3. Verify parquet + finalize + relative stats (arm relative, gripper absolute) + verify
        n_ep = 70
        t = np.linspace(0.0, 1.0, n_ep, dtype=np.float32)
        state = np.tile(np.arange(16, dtype=np.float32)[None, :], (n_ep, 1))
        state[:, :7] += t[:, None] * 0.5
        state[:, 7] = 100.0
        state[:, 15] = 25.0
        action = state.copy()
        action[:, :7] += 0.05

        write_episode_parquet(
            output_dir / "data" / "chunk-000" / "file-000.parquet",
            state=state,
            action=action,
            subtask_index=np.zeros(n_ep, dtype=np.int64),
            timestamp=(np.arange(n_ep, dtype=np.float32) / 30.0),
            frame_index=np.arange(n_ep, dtype=np.int64),
            episode_index=np.zeros(n_ep, dtype=np.int64),
            index=np.arange(n_ep, dtype=np.int64),
            task_index=np.zeros(n_ep, dtype=np.int64),
        )

        from types import SimpleNamespace

        dummy = SimpleNamespace(
            source_root=root,
            output_dir=output_dir,
            repo_id="local/selftest",
            robot_type="galbot_one_golf",
            fps=30,
            resample_uniform=True,
            target_resolution="224x224",
            resize_mode="pad",
            codec=DEFAULT_CODEC,
            crf=DEFAULT_CRF,
            gop=DEFAULT_GOP,
            preset=DEFAULT_PRESET,
            max_pad_mean=DEFAULT_MAX_PAD_MEAN,
            pts_tolerance_s=DEFAULT_PTS_TOLERANCE_S,
            skip_videos=False,
            chunk_size=30,
            arm_min_range=DEFAULT_ARM_MIN_RANGE,
            arm_min_std=DEFAULT_ARM_MIN_STD,
            include_padded_tail_in_stats=False,
            task_mode="unified",
            verify_samples=1,
            force=True,
        )
        prog = new_progress("selftest", 30)
        prog["total_episodes"] = 1
        prog["total_frames"] = n_ep
        prog["fps"] = 30
        prog["tasks"] = [DEFAULT_UNIFIED_TASK]
        prog["first_video_infos"] = {k: asdict(v) for k, v in vid_infos.items()}
        prog["first_height"] = 224
        prog["first_width"] = 224
        prog["raw_camera_shapes"] = {k: [480, 640] for k in CAMERA_TOPICS}
        prog["per_ep_video_stats"] = [ep_vid_stats]
        prog["episode_rows"] = [
            {
                "episode_index": 0,
                "tasks": [DEFAULT_UNIFIED_TASK],
                "length": n_ep,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": 0,
                "dataset_to_index": n_ep,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
                **{
                    f"videos/{ck}/{suffix}": val
                    for ck in CAMERA_TOPICS
                    for suffix, val in (
                        ("chunk_index", 0),
                        ("file_index", 0),
                        ("from_timestamp", 0.0),
                        ("to_timestamp", n_ep / 30.0),
                    )
                },
            }
        ]
        prog["conversion_entries"] = [
            {
                "episode_index": 0,
                "source_episode_id": "synthetic.SYNC.mcap",
                "frames": n_ep,
                "task": DEFAULT_UNIFIED_TASK,
            }
        ]

        run_finalize(dummy, prog)  # type: ignore[arg-type]
        rel_stats = json.loads((output_dir / "meta" / "stats.json").read_text(encoding="utf-8"))
        assert abs(rel_stats[ACTION]["mean"][7] - 100.0) < 1e-3, "left_gripper must stay absolute"
        assert abs(rel_stats[ACTION]["mean"][15] - 25.0) < 1e-3, "right_gripper must stay absolute"
        assert abs(rel_stats[ACTION]["mean"][4] - 4.0) > 1.0, "arm joints must be relative"
        assert abs(rel_stats[OBS_STATE]["mean"][4] - 4.25) < 0.1, "state stats must stay absolute"

        run_verify(dummy, prog)  # type: ignore[arg-type]

    print("SELF-TEST PASSED: 16-dim left-first + Causal ZOH + tail hold + gap split + "
          "libx264 crf=18 INTER_LINEAR + chunking + VLASH relative stats + verify OK.")


# ============================================================================
# 11. CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source-root", type=Path, default=_default_source_root())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--repo-id", type=str, default="local/dynamic_parcel_flip")
    parser.add_argument("--robot-type", type=str, default="galbot_one_golf")
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        help="Stages to run: 'all' or comma-separated subset of convert,finalize,verify.",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Uniform target FPS (0 = auto-detect).")
    parser.add_argument(
        "--no-resample-uniform",
        dest="resample_uniform",
        action="store_false",
        default=True,
        help="Disable uniform FPS resampling and use raw head_right camera frames.",
    )
    parser.add_argument("--target-resolution", type=str, default=DEFAULT_TARGET_RESOLUTION)
    parser.add_argument("--resize-mode", choices=["pad", "stretch"], default="pad")
    parser.add_argument("--codec", type=str, default=DEFAULT_CODEC)
    parser.add_argument("--crf", type=int, default=DEFAULT_CRF)
    parser.add_argument("--gop", type=int, default=DEFAULT_GOP)
    parser.add_argument("--preset", type=str, default=DEFAULT_PRESET)
    parser.add_argument("--no-video-verify", action="store_true")
    parser.add_argument("--max-first-frame-mae", type=float, default=DEFAULT_MAX_FIRST_FRAME_MAE)
    parser.add_argument("--max-pad-mean", type=float, default=DEFAULT_MAX_PAD_MEAN)
    parser.add_argument("--pts-tolerance-s", type=float, default=DEFAULT_PTS_TOLERANCE_S)
    parser.add_argument("--skip-videos", action="store_true")

    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES)
    parser.add_argument("--arm-tolerance-ms", type=float, default=DEFAULT_ARM_TOLERANCE_MS)
    parser.add_argument("--arm-startup-grace-s", type=float, default=DEFAULT_ARM_STARTUP_GRACE_S)
    parser.add_argument("--arm-startup-max-gap-ms", type=float, default=DEFAULT_ARM_STARTUP_MAX_GAP_MS)
    parser.add_argument("--hold-eps-rad", type=float, default=DEFAULT_HOLD_EPS_RAD)
    parser.add_argument("--disallow-missing-groups", action="store_true")
    parser.add_argument("--no-trim-leading-idle", action="store_true")
    parser.add_argument("--no-split-on-interior-gaps", action="store_true")

    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--arm-min-range", type=float, default=DEFAULT_ARM_MIN_RANGE)
    parser.add_argument("--arm-min-std", type=float, default=DEFAULT_ARM_MIN_STD)
    parser.add_argument(
        "--include-padded-tail-in-stats",
        action="store_true",
        help="Include clamped episode-tail actions in relative stats (off by default to match mask_action_pad=true).",
    )

    parser.add_argument(
        "--task-mode",
        choices=["unified", "per_folder"],
        default="unified",
        help="Default 'unified' assigns one goal prompt across all folders; 'per_folder' uses task_map.",
    )
    parser.add_argument("--unified-task", type=str, default=DEFAULT_UNIFIED_TASK)
    parser.add_argument("--default-task", type=str, default=DEFAULT_UNIFIED_TASK)
    parser.add_argument("--task-map-yaml", type=Path, default=None)

    parser.add_argument("--sync-only", action="store_true")
    parser.add_argument("--exclude-files", nargs="+", action="extend", default=[])
    parser.add_argument("--exclude-file-list", type=Path, default=None)
    parser.add_argument("--max-episodes-per-folder", type=int, default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--verify-samples", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes (1 = sequential).")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")

    args = parser.parse_args()
    args.allow_missing_groups = not args.disallow_missing_groups
    args.trim_leading_idle = not args.no_trim_leading_idle
    args.split_on_interior_gaps = not args.no_split_on_interior_gaps
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )

    if args.self_test:
        run_self_test()
        return

    steps = (
        ["convert", "finalize", "verify"]
        if args.step == "all"
        else [s.strip() for s in args.step.split(",") if s.strip()]
    )
    valid_steps = {"convert", "finalize", "verify"}
    unknown = [s for s in steps if s not in valid_steps]
    if unknown:
        raise ValueError(f"Unknown --step values {unknown}; expected subset of {sorted(valid_steps)}")

    task_map = dict(DEFAULT_TASK_MAP)
    if args.task_map_yaml is not None and args.task_map_yaml.exists():
        with args.task_map_yaml.open("r", encoding="utf-8") as f:
            task_map.update(yaml.safe_load(f) or {})

    exclude_patterns = load_exclude_patterns(args.exclude_files, args.exclude_file_list)

    if args.force and "convert" in steps and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fps_override = None if args.fps <= 0 else int(args.fps)
    fingerprint = compute_config_fingerprint(args, task_map, exclude_patterns)
    progress = load_progress(args.output_dir, fingerprint, fps_override)

    for step in steps:
        if step == "convert":
            run_convert(args, task_map, exclude_patterns, progress)
        elif step == "finalize":
            run_finalize(args, progress)
        elif step == "verify":
            run_verify(args, progress)


if __name__ == "__main__":
    main()
