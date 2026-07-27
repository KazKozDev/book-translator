#!/bin/bash

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# ---------------------------------------------------------------------------
# Logging — timestamped, leveled, colored from the interface palette
# ---------------------------------------------------------------------------
if [ -t 1 ] && [ -z "$NO_COLOR" ]; then
    C_RESET=$'\033[0m'
    C_BOLD=$'\033[1m'
    # 24-bit tones taken from :root in static/index.html, nudged where needed so
    # they stay legible on both light and dark terminal profiles.
    C_ACCENT=$'\033[38;2;91;124;153m'    # --accent #5b7c99 — headings, progress
    C_OK=$'\033[38;2;95;144;104m'        # muted sage, sits next to the accent
    C_WARN=$'\033[38;2;207;171;82m'      # #cfab52, the gold used in the UI
    C_ERR=$'\033[38;2;207;77;71m'        # between the UI's #b91c1c and #f87171
    C_MUTED=$'\033[38;2;138;143;152m'    # --muted #6b7280, lifted for dark bg
    C_CREAM=$'\033[38;2;240;214;170m'    # BOOK TRANSLATOR subtitle
else
    C_RESET=""; C_BOLD=""; C_ACCENT=""; C_OK=""; C_WARN=""; C_ERR=""
    C_MUTED=""; C_CREAM=""
fi

# Every log line reads "<mark> <message>", so continuation lines and prompts
# indent by that prefix to line up under the message column.
LOG_INDENT="  "

# _log <color> <mark> <message>
_log() {
    local color="$1" mark="$2"; shift 2
    printf '%s%s%s %s\n' "$color" "$mark" "$C_RESET" "$*"
}

log_step()   { printf '\n%s%s▸ %s%s\n' "$C_BOLD" "$C_ACCENT" "$*" "$C_RESET"; }
log_info()   { _log "$C_ACCENT" "·" "$*"; }
log_ok()     { _log "$C_OK"     "✓" "$*"; }
log_warn()   { _log "$C_WARN"   "!" "$*"; }
log_error()  { _log "$C_ERR"    "✗" "$*"; }
log_detail() { printf '%s%s%s%s\n' "$LOG_INDENT" "$C_MUTED" "$*" "$C_RESET"; }

fail() {
    log_error "$*"
    printf '\n'
    read -n 1 -s -r -p "$(printf '%s%sPress any key to close…%s' "$LOG_INDENT" "$C_MUTED" "$C_RESET")"
    printf '\n'
    exit 1
}

# ask <question> — yes/no prompt, aligned with the log column. Default is no.
ask() {
    local reply
    read -n 1 -s -r -p "$(printf '%s%s%s?%s %s %s[y/N]%s ' \
        "$LOG_INDENT" "$C_BOLD" "$C_WARN" "$C_RESET" "$*" "$C_MUTED" "$C_RESET")" reply
    printf '\n'
    [[ "$reply" =~ ^[Yy]$ ]]
}

# wait_for <seconds> <command…> — poll until the command succeeds, one dot per
# half-second, so a slow start shows progress without a line of log per try.
wait_for() {
    local timeout="$1"; shift
    "$@" >/dev/null 2>&1 && return 0
    printf '%s%s' "$LOG_INDENT" "$C_MUTED"
    local i
    for (( i = 1; i < timeout * 2; i++ )); do
        sleep 0.5
        printf '·'
        "$@" >/dev/null 2>&1 && { printf '%s\n' "$C_RESET"; return 0; }
    done
    printf '%s\n' "$C_RESET"
    return 1
}

# pip_install <label> <requirements file> — quiet on success, shows why on failure.
pip_install() {
    local label="$1" reqs="$2" out
    out=$(mktemp)
    log_info "Installing $label from $reqs …"
    if "$PYTHON" -m pip install -r "$reqs" >"$out" 2>&1; then
        rm -f "$out"
        return 0
    fi
    log_error "pip failed — its last lines:"
    tail -n 12 "$out" | while IFS= read -r line; do log_detail "$line"; done
    rm -f "$out"
    return 1
}

# Read the version straight from the header in the interface so the two never drift.
APP_VERSION=$(sed -n 's/.*class="brand-version">v\([0-9.]*\)<.*/\1/p' static/index.html 2>/dev/null | head -1)
[ -n "$APP_VERSION" ] || APP_VERSION="3.0"

print_logo() {
    printf '\n\n\n%s%s' "$C_BOLD" "$C_ACCENT"
    cat << 'LOGO1'
████████╗ ██████╗ ██╗     ███╗   ███╗ █████╗  ██████╗██╗  ██╗
╚══██╔══╝██╔═══██╗██║     ████╗ ████║██╔══██╗██╔════╝██║  ██║
   ██║   ██║   ██║██║     ██╔████╔██║███████║██║     ███████║
   ██║   ██║   ██║██║     ██║╚██╔╝██║██╔══██║██║     ██╔══██║
   ██║   ╚██████╔╝███████╗██║ ╚═╝ ██║██║  ██║╚██████╗██║  ██║
   ╚═╝    ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
LOGO1
    printf '%s' "$C_RESET"
    # Centered under the 61-column TOLMACH block above.
    local subtitle="B O O K   T R A N S L A T O R  v$APP_VERSION"
    printf '\n%s%s%*s%s%s\n\n\n' \
        "$C_BOLD" "$C_CREAM" $(( (61 - ${#subtitle}) / 2 )) "" "$subtitle" "$C_RESET"
}

print_logo
log_detail "$(date '+%a %d %b %Y, %H:%M') · $(sw_vers -productName) $(sw_vers -productVersion)"
log_detail "$DIR"

# ---------------------------------------------------------------------------
# Python 3
# ---------------------------------------------------------------------------
log_step "Python"
if ! command -v python3 >/dev/null 2>&1; then
    fail "Python 3 is not installed. Install it from https://www.python.org/downloads/ and run this again."
fi
log_ok "$(python3 --version 2>&1)"
log_detail "$(command -v python3)"

# Keep dependencies isolated from Homebrew's externally managed Python.
VENV_DIR="$DIR/venv"
PYTHON="$VENV_DIR/bin/python"

log_step "Virtual environment"
if ! "$PYTHON" -c "import sys" >/dev/null 2>&1; then
    log_info "Creating it (first run) …"
    if ! python3 -m venv --clear "$VENV_DIR"; then
        fail "Could not create the virtual environment at $VENV_DIR."
    fi
    log_ok "Created"
else
    log_ok "Present"
fi
log_detail "$VENV_DIR"

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
log_step "Dependencies"
if ! "$PYTHON" -c "import flask, flask_cors, psutil, gliner, rapidfuzz, sentence_transformers" >/dev/null 2>&1; then
    pip_install "dependencies" requirements.txt \
        || fail "Fix the error above, then run this again."
    log_ok "Installed"
else
    log_ok "Present"
fi
DEP_VERSIONS=$("$PYTHON" -c "
from importlib.metadata import version
print(' · '.join(f'{p} {version(p)}' for p in ('flask', 'flask-cors', 'psutil')))
" 2>/dev/null)
[ -n "$DEP_VERSIONS" ] && log_detail "$DEP_VERSIONS"

# Optional COMET-Kiwi quality estimation for the Tests panel. Off by default:
# it pulls in torch and downloads a multi-GB checkpoint on first use.
if [ -f "requirements-eval.txt" ] && ! "$PYTHON" -c "import comet" >/dev/null 2>&1; then
    log_step "Quality-estimation extras (optional)"
    log_detail "COMET-Kiwi powers the Tests panel · several GB, torch included"
    if ask "Install them now"; then
        if pip_install "COMET extras" requirements-eval.txt; then
            log_ok "Installed"
        else
            log_warn "Skipping — the Tests panel stays unavailable until this succeeds."
        fi
    else
        log_info "Skipped — the Tests panel stays unavailable"
    fi
fi

# LaBSE alignment and multilingual Language ID are separate quality tools.
# Unlike the core translator they download large checkpoints only when their
# respective Run button is pressed, so leave installation as an explicit
# choice at launch time.
if [ -f "requirements-quality.txt" ] && ! "$PYTHON" -c "import sentence_transformers, transformers" >/dev/null 2>&1; then
    log_step "Document-quality extras (optional)"
    log_detail "LaBSE alignment + multilingual Language ID · model weights download on first Run"
    if ask "Install them now"; then
        if pip_install "document-quality extras" requirements-quality.txt; then
            log_ok "Installed"
        else
            log_warn "Skipping — LaBSE and Language ID will explain how to enable themselves."
        fi
    else
        log_info "Skipped — install later to enable LaBSE and Language ID"
    fi
fi

# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------
OLLAMA_API="http://localhost:11434/api/tags"
ollama_up() { curl -s -m 2 "$OLLAMA_API" >/dev/null 2>&1; }

log_step "Ollama"
if ! command -v ollama >/dev/null 2>&1; then
    log_error "Ollama is not installed. Opening the download page …"
    open "https://ollama.com/download"
    fail "Install Ollama, pull a model, then run this again."
fi
log_ok "Found at $(command -v ollama)"

if ollama_up; then
    log_ok "Server already running on port 11434"
else
    log_info "Server not running — starting it …"
    ollama serve >/dev/null 2>&1 &
    if ! wait_for 10 ollama_up; then
        fail "Could not start Ollama. Start it manually with 'ollama serve' and run this again."
    fi
    log_ok "Server up on port 11434"
fi

MODELS=$(curl -s -m 3 "$OLLAMA_API" 2>/dev/null | grep -o '"name":"[^"]*"' | cut -d'"' -f4)
MODEL_COUNT=$(printf '%s' "$MODELS" | grep -c .)
if [ "$MODEL_COUNT" -eq 0 ]; then
    log_warn "No models pulled yet — translation will fail until you pull one"
    log_detail "e.g. ollama pull qwen2.5:7b"
else
    log_ok "$MODEL_COUNT model(s) available locally"
    printf '%s\n' "$MODELS" | head -5 | while IFS= read -r model; do log_detail "$model"; done
    [ "$MODEL_COUNT" -gt 5 ] && log_detail "… and $((MODEL_COUNT - 5)) more"
fi

# ---------------------------------------------------------------------------
# Port
# ---------------------------------------------------------------------------
PORT=5001
log_step "Port $PORT"
EXISTING_PIDS=$(lsof -ti:$PORT 2>/dev/null)
if [ -n "$EXISTING_PIDS" ]; then
    log_warn "In use by PID $(echo "$EXISTING_PIDS" | tr '\n' ' ')— terminating"
    echo "$EXISTING_PIDS" | xargs kill -9 2>/dev/null
    sleep 1
    log_ok "Freed"
else
    log_ok "Free"
fi

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
log_step "Translation cache"
if [ -f cache.db ]; then
    CACHE_SIZE=$(du -h cache.db | cut -f1 | tr -d ' ')
    rm -f cache.db
    log_ok "Cleared cache.db ($CACHE_SIZE)"
else
    log_ok "Already empty"
fi

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
URL="http://localhost:$PORT"
port_listening() { lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; }

log_step "Server"
log_info "Starting translator.py …"
TOLMACH_BANNER_PRINTED=1 "$PYTHON" translator.py &
SERVER_PID=$!
STARTED_AT=$SECONDS

if ! wait_for 15 port_listening; then
    fail "translator.py did not open port $PORT — see its output above for the reason."
fi
log_ok "Listening on $PORT after $((SECONDS - STARTED_AT))s"

HTTP_CODE=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$URL/" 2>/dev/null)
if [ "$HTTP_CODE" = "200" ]; then
    log_ok "Interface responding"
else
    log_warn "Port is open but $URL/ answered HTTP ${HTTP_CODE:-nothing} — the page may not load"
fi

# The unique query string stops the browser from re-showing an already open tab
# with a stale copy of the interface.
open "$URL/?launched=$(date +%s)"
log_ok "Opened in your browser"

log_step "Ready"
log_ok "Book Translator v$APP_VERSION · $URL"
log_detail "server PID $SERVER_PID · ready in ${SECONDS}s"
printf '\n%s%sPress Ctrl+C to stop the server%s\n\n' "$C_BOLD" "$C_ACCENT" "$C_RESET"

wait $SERVER_PID
