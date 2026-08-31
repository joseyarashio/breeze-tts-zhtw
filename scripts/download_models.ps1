# Download both checkpoints with the Hugging Face CLI (pip install -U huggingface_hub).
# Breeze-TTS-2 weights are research / non-commercial — accept the terms on HF first.
param(
  [string]$Root = "C:\ai-models"
)
$ErrorActionPreference = "Stop"
hf download BreezeBlue/Breeze-TTS-2 --local-dir "$Root\breeze-tts-2"
hf download MediaTek-Research/Breeze-ASR-25 --local-dir "$Root\breeze-asr"
python "$PSScriptRoot\patch_checkpoint.py" "$Root\breeze-tts-2"
Write-Host "done → $Root\breeze-tts-2 , $Root\breeze-asr"
