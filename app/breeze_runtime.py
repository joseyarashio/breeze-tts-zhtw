"""Model loading for Breeze TTS 2 — our replacement for upstream's
``breeze_infer.runtime.load_runtime`` so the submodule stays untouched.

Identical to upstream except for the Windows workaround: transformers'
meta-device / mmap shard loading (``get_slice`` + ``param[...]`` over
``UntypedStorage.from_file``) segfaults with an access violation on Windows
(seen on torch 2.9.1 and 2.11.0, safetensors 0.7.0 and 0.8.0, for both the
TTS shards and the single-file Whisper checkpoint). Reading tensors with
``safe_open(...).get_tensor`` works, so on Windows we preload the full state
dict and call ``from_pretrained(None, config=..., state_dict=...)``.
Linux (Docker) uses the plain upstream path.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer

from breeze_infer.runtime import get_dist_info  # upstream (submodule)
from models.breeze import BreezeForConditionalGeneration  # upstream (submodule)


def safe_state_dict_load_needed() -> bool:
    """True on Windows (mmap shard loading segfaults) or when forced via env."""
    forced = os.environ.get("BREEZE_SAFE_LOAD")
    if forced is not None:
        return forced == "1"
    return sys.platform == "win32"


def load_safetensors_state_dict(directory: Path, pattern: str = "*.safetensors") -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    state_dict: dict[str, torch.Tensor] = {}
    for shard in sorted(glob.glob(str(Path(directory) / pattern))):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def load_runtime(
    ckpt_dir: Path,
    *,
    device: str,
    attn_implementation: str,
) -> tuple[AutoTokenizer, BreezeForConditionalGeneration, Any]:
    ckpt_dir = Path(ckpt_dir)
    if device.startswith("cuda"):
        try:
            torch.cuda.set_device(device)
        except Exception as exc:
            rank, world_size, local_rank = get_dist_info()
            raise RuntimeError(
                "Failed to set CUDA device "
                f"device={device} rank={rank} world_size={world_size} local_rank={local_rank} "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
                f"device_count={torch.cuda.device_count()}"
            ) from exc

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    if safe_state_dict_load_needed():
        state_dict = load_safetensors_state_dict(ckpt_dir, "model-*.safetensors")
        config = AutoConfig.from_pretrained(ckpt_dir)
        model = BreezeForConditionalGeneration.from_pretrained(
            None,
            config=config,
            state_dict=state_dict,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )
        del state_dict
    else:
        model = BreezeForConditionalGeneration.from_pretrained(
            ckpt_dir,
            dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )
    model.to(device).eval()

    from qwen_tts import Qwen3TTSTokenizer

    bundled_audio_tokenizer = ckpt_dir / "audio_tokenizer"
    if not bundled_audio_tokenizer.is_dir():
        raise FileNotFoundError(
            "Bundled audio tokenizer not found at "
            f"{bundled_audio_tokenizer}. The Breeze model package must include "
            "the audio_tokenizer directory."
        )
    audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
        str(bundled_audio_tokenizer), device_map=device
    )
    return tokenizer, model, audio_tokenizer
