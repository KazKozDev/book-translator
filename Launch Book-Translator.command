#!/bin/bash
# macOS double-click entry point.  The actual bootstrap is Python so the same
# setup logic is shared with Linux and Windows.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
exec python3 "$DIR/launch.py"
