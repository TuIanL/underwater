#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ANNOTATION_ROOT="${1:-data/annotations/seed}"
FRAME_ROOT="${2:-data/frames}"

UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.uv-cache}" \
uv run --project "$REPO_ROOT" --directory "$REPO_ROOT" swim-pose annotations web \
  --annotation-root "$ANNOTATION_ROOT" \
  --frame-root "$FRAME_ROOT"
