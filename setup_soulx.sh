#!/bin/bash
# One-time setup for SoulX-FlashHead Lite on RunPod.
# Run AFTER setup.sh (main LiveTalking deps must already be installed).
set -e

VENV=/workspace/soulx_venv
REPO=/workspace/SoulX-FlashHead
MODELS=/workspace/models

echo "=== [1/5] Creating isolated Python venv for SoulX ==="
python3 -m venv "$VENV"

echo "=== [2/5] Cloning SoulX-FlashHead ==="
if [ ! -d "$REPO" ]; then
    git clone https://github.com/Soul-AILab/SoulX-FlashHead "$REPO"
else
    echo "  already cloned, skipping"
fi

echo "=== [3/5] Installing SoulX dependencies ==="
"$VENV/bin/pip" install --upgrade pip --quiet
# Torch first (cu128 to match RunPod driver)
"$VENV/bin/pip" install \
    "torch==2.9.1+cu128" "torchvision==0.24.1+cu128" "torchaudio==2.9.1+cu128" \
    --index-url https://download.pytorch.org/whl/cu128 --quiet
# SoulX requirements — patch version pins that conflict with cu128 torch
sed \
  -e 's/mediapipe==0\.10\.9/mediapipe>=0.10.13/' \
  -e '/nvidia-nccl-cu12/d' \
  -e '/xformers/d' \
  "$REPO/requirements.txt" > /tmp/soulx_req_patched.txt
"$VENV/bin/pip" install -r /tmp/soulx_req_patched.txt --quiet
# Install xformers separately (no strict nccl pin)
"$VENV/bin/pip" install xformers --quiet
# Install the SoulX package itself
"$VENV/bin/pip" install -e "$REPO" --quiet

echo "=== [4/5] Downloading wav2vec2-base-960h ==="
mkdir -p "$MODELS"
"$VENV/bin/python" - <<'EOF'
from huggingface_hub import snapshot_download
import os
dest = os.environ.get('MODELS', '/workspace/models')
snapshot_download('facebook/wav2vec2-base-960h', local_dir=f'{dest}/wav2vec2-base-960h')
print('wav2vec2 downloaded')
EOF

echo "=== [5/5] Downloading SoulX-FlashHead-1_3B ==="
"$VENV/bin/python" - <<'EOF'
from huggingface_hub import snapshot_download
import os
dest = os.environ.get('MODELS', '/workspace/models')
snapshot_download('Soul-AILab/SoulX-FlashHead', local_dir=f'{dest}/SoulX-FlashHead-1_3B')
print('SoulX model downloaded')
EOF

echo ""
echo "=== SoulX setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Place your portrait photo at: data/avatars/my_avatar/source.jpg"
echo "  2. Set model: soulx in config.yaml"
echo "  3. Run: PYTHONPATH=/workspace/live-talking python app.py"
