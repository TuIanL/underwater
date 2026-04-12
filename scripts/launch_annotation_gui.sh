#!/bin/zsh

set -euo pipefail

ANNOTATION_ROOT="${1:-data/annotations/seed}"
FRAME_ROOT="${2:-data/frames}"

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src \
/usr/bin/python3 -m swim_pose.cli annotations web \
  --annotation-root "$ANNOTATION_ROOT" \
  --frame-root "$FRAME_ROOT"
