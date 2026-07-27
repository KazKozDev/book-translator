#!/usr/bin/env bash
# Linux/macOS terminal entry point.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
exec python3 "$DIR/launch.py"
