#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /absolute/path/to/video.mp4 [extra args...]" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv_orin"
VIDEO_PATH="$1"
shift

if [ ! -x "${VENV_DIR}/bin/python" ]; then
  echo "Missing ${VENV_DIR}/bin/python. Run scripts/setup_orin_nano.sh first." >&2
  exit 1
fi

cd "${REPO_ROOT}"

"${VENV_DIR}/bin/python" auto_select_truck_frames.py "${VIDEO_PATH}" --device auto --save-frames 0 "$@"
