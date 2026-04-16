#!/usr/bin/env bash
set -euo pipefail

# Packages the current project tree for upload or sync into Google Colab.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/dist}"
ARCHIVE_PATH="$OUTPUT_DIR/nfse-extractor-colab.tar.gz"

mkdir -p "$OUTPUT_DIR"

tar -czf "$ARCHIVE_PATH" \
  -C "$ROOT_DIR" \
  .codex \
  configs \
  docs \
  notebooks \
  scripts \
  src \
  tests \
  .gitignore \
  pyproject.toml \
  requirements.txt \
  requirements-dev.txt \
  README.md

printf 'Created %s\n' "$ARCHIVE_PATH"
