#!/usr/bin/env bash
set -euo pipefail

# Bootstraps a Google Colab runtime for this project.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

apt-get update
apt-get install -y tesseract-ocr

python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/requirements.txt"

printf 'Colab bootstrap completed in %s\n' "$ROOT_DIR"
