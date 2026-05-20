#!/usr/bin/env bash
set -euo pipefail

# Bootstraps a Google Colab runtime for this project.
#
# Usage:
#   bash colab_bootstrap.sh            # Tesseract only (default)
#   bash colab_bootstrap.sh --dolphin  # Tesseract + Dolphin-v2 GPU deps

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Parse flags ───────────────────────────────────────────────────────────────
INSTALL_DOLPHIN=0
for arg in "$@"; do
    if [[ "$arg" == "--dolphin" ]]; then
        INSTALL_DOLPHIN=1
    fi
done

# ── System packages ───────────────────────────────────────────────────────────
apt-get update -qq
apt-get install -y tesseract-ocr tesseract-ocr-por

# ── Base Python dependencies ──────────────────────────────────────────────────
python -m pip install --upgrade pip
cd "$ROOT_DIR"
python -m pip install -r requirements.txt

# ── Optional: Dolphin-v2 GPU dependencies ────────────────────────────────────
if [[ $INSTALL_DOLPHIN -eq 1 ]]; then
    printf '\nInstalling Dolphin-v2 dependencies…\n'
    python -m pip install \
        "torch>=2.6.0" \
        "transformers>=4.51.0" \
        "accelerate>=1.4.0" \
        "qwen_vl_utils" \
        "opencv-python" \
        "numpy>=1.24"
    printf 'Dolphin-v2 dependencies installed.\n'
fi

printf 'Colab bootstrap completed in %s\n' "$ROOT_DIR"
