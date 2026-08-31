"""Patch a downloaded Breeze-TTS-2 checkpoint for machines without flash-attn.

The checkpoint's config.json sets
    text_encoder_config.preferred_attn_implementation = "flash_attention_2"
and upstream's T5Gemma2 text encoder honours that key regardless of the
attn_implementation you pass to from_pretrained. On Windows (no flash-attn
wheel) model load fails with an ImportError. This switches it to "sdpa".
Idempotent; safe to run on Linux too (sdpa is fine either way).

    python scripts/patch_checkpoint.py C:\\ai-models\\breeze-tts-2
"""

import json
import sys
from pathlib import Path


def main() -> None:
    ckpt = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\ai-models\breeze-tts-2")
    path = ckpt / "config.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    patched = []

    def walk(node: dict, prefix: str = "") -> None:
        for key, value in node.items():
            if key == "preferred_attn_implementation" and value == "flash_attention_2":
                node[key] = "sdpa"
                patched.append(prefix + key)
            elif isinstance(value, dict):
                walk(value, prefix + key + ".")

    walk(cfg)
    if patched:
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        print("patched:", ", ".join(patched))
    else:
        print("nothing to patch (already sdpa/eager)")


if __name__ == "__main__":
    main()
