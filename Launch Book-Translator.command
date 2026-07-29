#!/bin/bash
# macOS double-click entry point.  The actual bootstrap is Python so the same
# setup logic is shared with Linux and Windows.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
# A .command file always opens in Terminal, where the Tolmach banner is meant
# to carry its blue/cream identity. Codex and some shell profiles export
# NO_COLOR globally; override that for this launcher banner only.
export TOLMACH_FORCE_BANNER_COLOR=1
exec python3 "$DIR/launch.py"
