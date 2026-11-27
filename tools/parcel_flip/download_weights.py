#!/usr/bin/env python3
"""Resumable weight downloader for InternVLA-A1.5 + VLASH offline training.

Downloads:
  - InternRobotics/InternVLA-A1.5-base -> /mnt/data3/InternVLA-A1.5-base
  - Wan-AI/Wan2.2-TI2V-5B              -> /mnt/data3/Wan2.2-TI2V-5B
  - Qwen/Qwen3.5-2B                    -> /mnt/data3/Qwen3.5-2B
  - physical-intelligence/fast         -> /mnt/data3/fast_tokenizer

Supports ModelScope (default, fast in mainland China) and HuggingFace Mirror.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


MODELS = [
    {
        "name": "InternVLA-A1.5-base",
        "hf_id": "InternRobotics/InternVLA-A1.5-base",
        "ms_id": "InternRobotics/InternVLA-A1.5-base",
        "local_dir": "/mnt/data3/InternVLA-A1.5-base",
    },
    {
        "name": "Wan2.2-TI2V-5B",
        "hf_id": "Wan-AI/Wan2.2-TI2V-5B",
        "ms_id": "Wan-AI/Wan2.2-TI2V-5B",
        "local_dir": "/mnt/data3/Wan2.2-TI2V-5B",
        "ignore_patterns": ["models_t5*", "google/*", "assets/*", "examples/*"],
    },
    {
        "name": "Qwen3.5-2B",
        "hf_id": "Qwen/Qwen3.5-2B",
        "ms_id": "Qwen/Qwen3.5-2B",
        "local_dir": "/mnt/data3/Qwen3.5-2B",
    },
    {
        "name": "fast_tokenizer",
        "hf_id": "physical-intelligence/fast",
        "ms_id": None,
        "local_dir": "/mnt/data3/fast_tokenizer",
    },
]


def is_model_complete(entry: dict, local_dir: Path) -> bool:
    if not local_dir.exists() or any(local_dir.glob("*.incomplete")):
        return False
    name = entry["name"]
    if name == "InternVLA-A1.5-base":
        return (local_dir / "model.safetensors").is_file() and (local_dir / "config.json").is_file()
    if name == "Wan2.2-TI2V-5B":
        return (
            (local_dir / "Wan2.2_VAE.pth").is_file()
            and (local_dir / "config.json").is_file()
            and len(list(local_dir.glob("diffusion_pytorch_model*.safetensors"))) >= 3
        )
    if name == "Qwen3.5-2B":
        if not ((local_dir / "config.json").is_file() and (local_dir / "tokenizer.json").is_file()):
            return False
        index_file = local_dir / "model.safetensors.index.json"
        if index_file.is_file():
            with index_file.open("r", encoding="utf-8") as f:
                weight_map = json.load(f).get("weight_map", {})
            shards = sorted(set(weight_map.values()))
            return bool(shards) and all((local_dir / shard).is_file() for shard in shards)
        return len(list(local_dir.glob("*.safetensors"))) >= 1
    if name == "fast_tokenizer":
        return (
            (local_dir / "tokenizer.json").is_file()
            and (local_dir / "processing_action_tokenizer.py").is_file()
        )
    return any(local_dir.glob("*.safetensors"))


def download_one(
    entry: dict,
    source: str,
    cache_dir: Path,
    skip_existing: bool = False,
) -> None:
    local_dir = Path(entry["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)
    if skip_existing and is_model_complete(entry, local_dir):
        print(f"[SKIP] {entry['name']} already complete in {local_dir}")
        return

    ignore_patterns = entry.get("ignore_patterns")
    use_ms = source == "modelscope" and entry["ms_id"] is not None
    if use_ms:
        print(f"[ModelScope] Downloading {entry['ms_id']} -> {local_dir}")
        try:
            from modelscope.hub.snapshot_download import snapshot_download as ms_download

            ms_download(
                model_id=entry["ms_id"],
                local_dir=str(local_dir),
                cache_dir=str(cache_dir / "modelscope"),
                ignore_file_pattern=ignore_patterns,
            )
            return
        except Exception as exc:
            print(f"[ModelScope] Failed for {entry['name']}: {exc}; falling back to HF mirror.")

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    from huggingface_hub import snapshot_download as hf_download

    print(f"[HF Mirror: {os.environ['HF_ENDPOINT']}] Downloading {entry['hf_id']} -> {local_dir}")
    hf_download(
        repo_id=entry["hf_id"],
        local_dir=str(local_dir),
        cache_dir=str(cache_dir / "huggingface"),
        ignore_patterns=ignore_patterns,
    )


def verify_safetensors(root: Path) -> None:
    from safetensors import safe_open

    shards = sorted(root.rglob("*.safetensors"))
    if not shards:
        print(f"[WARN] No .safetensors files found under {root}")
        return
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            keys = list(f.keys())
        print(f"[OK] {shard.name}: {len(keys)} tensors")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["modelscope", "hf"],
        default="modelscope",
        help="Primary download source (falls back to hf-mirror if unavailable).",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of model names (InternVLA-A1.5-base, Wan2.2-TI2V-5B, Qwen3.5-2B, fast_tokenizer).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/mnt/data3/jiangxirui/.cache/hub"),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--keep-proxy",
        action="store_true",
        help="Keep http_proxy/https_proxy env vars (disabled by default for direct domestic speed).",
    )
    args = parser.parse_args()

    if not args.keep_proxy:
        for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
            os.environ.pop(proxy_var, None)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    selected = [m for m in MODELS if not args.only or m["name"] in args.only]

    if not args.verify_only:
        for entry in selected:
            download_one(entry, source=args.source, cache_dir=args.cache_dir, skip_existing=args.skip_existing)

    for entry in selected:
        local_dir = Path(entry["local_dir"])
        if local_dir.exists():
            print(f"=== Verifying {entry['name']} ({local_dir}) ===")
            verify_safetensors(local_dir)


if __name__ == "__main__":
    main()
