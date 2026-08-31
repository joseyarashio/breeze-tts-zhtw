# breeze-tts-zhtw — Breeze TTS 2 for Traditional Chinese

A 繁體中文-first wrapper around [BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2)
(open-weight zh/en TTS with voice clone, voice design and inline vocal events).

What this adds on top of upstream:

- **Web UI** — paste a paragraph, drop a reference clip, preview, download WAV.
- **繁 → 簡 before synthesis** — type Traditional, we convert with OpenCC (live preview of
  what will be spoken). Kept ON by default: in A/B listening tests on the same seed the
  Simplified-fed output was clearly better (an ASR round-trip shows no difference, so judge by ear). Applies to vocal-event
  tags too (`[嘆氣]` → `[叹气]`).
- **Reference clip auto-transcription** — upload a clip with no transcript and the
  逐字稿 field is filled by Breeze-ASR-25 (Whisper-large-v2 fine-tune, Traditional output),
  either via an external ASR service or in-process.
- **Vocal-event chips** — one click inserts `[笑]`, `(sigh)`, … at the cursor.
- **Standalone Docker image** — one container, GPU, weights mounted read-only.
- **Windows-native fallback** — works around two Windows-only problems (see below).

Upstream lives in `upstream/breeze-tts` as a git submodule and is **never modified**;
everything of ours is under `app/`, `scripts/` and the Docker files.

## Layout

```
app/server.py           FastAPI server (UI + /api/tts + /api/transcribe_ref + /api/convert)
app/breeze_runtime.py   our load_runtime (replaces upstream's; adds Windows-safe loading)
app/ui/index.html       single-file UI
scripts/                download_models.ps1, patch_checkpoint.py
upstream/breeze-tts     git submodule → breezeblue-ai/breeze-tts (pinned)
Dockerfile, docker-compose.yml
```

## Quick start (Docker, recommended)

```bash
git clone --recurse-submodules <this repo>
cd breeze-tts-zhtw
# weights (≈7.2 GB + 3.1 GB) go outside the repo, default C:\ai-models\{breeze-tts-2,breeze-asr}
pwsh scripts/download_models.ps1          # or set BREEZE_TTS2_WEIGHTS / BREEZE_ASR_WEIGHTS
docker compose up -d --build              # first build ≈ 10 min, image ≈ 13.5 GB
```

Open http://localhost:7772 (`BREEZE_PORT` to change the host port).
`/health` reports device, VRAM, and whether the in-process ASR is loaded.

Requirements: NVIDIA GPU ≥ 12 GB (7.2 GB resident + 3.1 GB while ASR is loaded),
Docker with the NVIDIA container runtime, CUDA 12.8-capable driver (RTX 50-series OK).

## Windows-native run (no Docker)

```powershell
py -3.10 -m venv .venv
.venv\Scripts\pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -r upstream\breeze-tts\requirements.txt -r requirements.txt
python scripts\patch_checkpoint.py C:\ai-models\breeze-tts-2
.venv\Scripts\python app\server.py       # http://localhost:7772
```

Two Windows-only issues are handled automatically:

1. The checkpoint's `config.json` asks the text encoder for `flash_attention_2`
   (no Windows wheel) → `scripts/patch_checkpoint.py` switches it to `sdpa`.
2. transformers' mmap shard loading segfaults on Windows (torch 2.9/2.11, safetensors
   0.7/0.8) → `app/breeze_runtime.py` preloads the state dict via `safe_open().get_tensor`
   when `sys.platform == "win32"` (force with `BREEZE_SAFE_LOAD=1/0`).

## Configuration (env)

| Variable | Default | Meaning |
|---|---|---|
| `BREEZE_TTS2_MODEL_PATH` | `C:\ai-models\breeze-tts-2` | TTS weights (`/models/breeze-tts-2` in Docker) |
| `BREEZE_ASR_MODEL_PATH` | `C:\ai-models\breeze-asr` | Breeze-ASR-25 for the in-process fallback |
| `BREEZE_ASR_SERVICE_URL` | `""` | External Breeze ASR base URL (e.g. `http://127.0.0.1:8765`). Empty = always in-process |
| `BREEZE_ASR_IDLE_UNLOAD_SEC` | `600` | Unload the in-process ASR after idle |
| `BREEZE_TTS2_PORT` / `HOST` | `7772` / `0.0.0.0` | Bind |
| `BREEZE_TTS2_FAST` | `0` | Upstream fast path (Linux + flash-attn wheel, ~14.4 GB VRAM) |

## API

- `POST /api/tts` — form: `text`, `ref_audio` (file, optional), `ref_text`, `instruction`,
  `convert_t2s` (`1`/`0`), `cfg_scale`, `seed` → `{url, duration_sec, rtf, spoken_text, …}`
- `POST /api/transcribe_ref` — form: `ref_audio` → `{text, source: "asr-service"|"local"}`
- `POST /api/convert` — form: `text` → `{converted, changed}`
- `GET /api/audio/<id>.wav?download=1`

## Notes

- Vocal events documented upstream: `[笑] [叹气] [咳嗽] [清嗓子]` / `(laugh) (sigh) (cough) (clears throat)`.
  They are plain text, not tokens; other tags are experimental.
- `cfg_scale` 1.0 for plain cloning; upstream recommends 4 for voice design / voice direction.
- Measured speed without the fast path: RTF ≈ 2.2–2.7 on an RTX 5070 Ti (upstream's 0.32 needs flash-attn + 24 GB).
- Licensing: upstream code Apache-2.0; **Breeze-TTS-2 weights are research / non-commercial** — that is why they are never baked into the image.
