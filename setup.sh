#!/bin/bash
# Run this once after every RunPod pod restart to reinstall dependencies.
set -e

cd /workspace/live-talking

# Step 1: Install requirements (may install wrong torch version)
pip install -r requirements.txt --ignore-installed blinker cryptography

# Step 2: Pin versions that requirements.txt gets wrong
pip install "transformers==4.46.3" "diffusers==0.31.0" "huggingface-hub==0.26.5" "tokenizers==0.20.3" --force-reinstall

# Step 3: Force correct torch versions LAST (requirements.txt overwrites these)
pip install "torch==2.9.1+cu128" "torchvision==0.24.1+cu128" "torchaudio==2.9.1+cu128" \
    --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

# Step 4: face_recognition (needs TMPDIR on /workspace due to disk space)
TMPDIR=/workspace pip install face_recognition

echo ""
echo "Setup complete. Run: PYTHONPATH=/workspace/live-talking python app.py"
