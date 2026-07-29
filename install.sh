#!/bin/sh
# Install Tolmach for the current macOS or Linux user.
#
# This script only delivers the application and a suitable Python interpreter.
# The installed `tolmach` command runs launch.py in a real terminal, where the
# existing bootstrap can ask before installing Ollama or downloading models.
set -eu

REPOSITORY="${TOLMACH_REPOSITORY:-KazKozDev/book-translator}"
REF="${TOLMACH_REF:-main}"
ARCHIVE_URL="${TOLMACH_ARCHIVE_URL:-https://github.com/${REPOSITORY}/archive/refs/heads/${REF}.tar.gz}"
UV_INSTALL_URL="${TOLMACH_UV_INSTALL_URL:-https://astral.sh/uv/0.11.32/install.sh}"

INSTALL_ROOT="${HOME}/.local/share/tolmach"
BIN_DIR="${HOME}/.local/bin"
TOOLS_DIR="${INSTALL_ROOT}/tools"
DATA_DIR="${INSTALL_ROOT}/data"
RELEASES_DIR="${INSTALL_ROOT}/releases"

fail() {
    printf 'Tolmach installer: %s\n' "$*" >&2
    exit 1
}

case "$(uname -s)" in
    Darwin|Linux) ;;
    *) fail "this installer supports macOS and Linux; use Launch Book-Translator.bat on Windows." ;;
esac

command -v curl >/dev/null 2>&1 || fail "curl is required."
command -v tar >/dev/null 2>&1 || fail "tar is required."

python_is_supported() {
    "$1" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))' >/dev/null 2>&1
}

find_python() {
    if [ -n "${TOLMACH_PYTHON:-}" ]; then
        explicit_python="$(command -v "${TOLMACH_PYTHON}" 2>/dev/null || true)"
        [ -n "$explicit_python" ] ||
            fail "TOLMACH_PYTHON does not name an executable interpreter."
        python_is_supported "$explicit_python" ||
            fail "TOLMACH_PYTHON must point to Python 3.10 or later."
        printf '%s\n' "$explicit_python"
        return
    fi

    if [ "${TOLMACH_FORCE_UV:-0}" != "1" ]; then
        for candidate in python3 python; do
            if command -v "$candidate" >/dev/null 2>&1; then
                candidate_path="$(command -v "$candidate")"
                if python_is_supported "$candidate_path"; then
                    printf '%s\n' "$candidate_path"
                    return
                fi
            fi
        done
    fi

    mkdir -p "$TOOLS_DIR"
    uv_bin="${TOOLS_DIR}/uv"
    if [ ! -x "$uv_bin" ]; then
        printf 'Installing uv so Tolmach can use Python 3.12...\n' >&2
        curl -fsSL "$UV_INSTALL_URL" |
            env UV_UNMANAGED_INSTALL="$TOOLS_DIR" sh >&2
    fi
    [ -x "$uv_bin" ] || fail "uv installation did not produce ${uv_bin}."

    "$uv_bin" python install 3.12 >&2
    managed_python="$("$uv_bin" python find 3.12)"
    python_is_supported "$managed_python" ||
        fail "uv did not provide a usable Python 3.12 interpreter."
    printf '%s\n' "$managed_python"
}

python_bin="$(find_python)"

temporary_dir="$(mktemp -d "${TMPDIR:-/tmp}/tolmach-install.XXXXXX")"
cleanup() {
    rm -rf "$temporary_dir"
}
trap cleanup 0 HUP INT TERM

archive="${temporary_dir}/tolmach.tar.gz"
source_dir="${temporary_dir}/source"
printf 'Downloading Tolmach from %s@%s...\n' "$REPOSITORY" "$REF"
curl -fsSL "$ARCHIVE_URL" -o "$archive"

# Refuse absolute paths and parent traversal before tar writes anything.
if ! tar -tzf "$archive" | awk '
    /^\// { unsafe = 1 }
    /(^|\/)\.\.($|\/)/ { unsafe = 1 }
    END { exit unsafe ? 1 : 0 }
'; then
    fail "the downloaded archive contains unsafe paths."
fi

mkdir -p "$source_dir"
tar -xzf "$archive" -C "$source_dir" --strip-components=1

for required in launch.py requirements.txt src/translator.py; do
    [ -f "${source_dir}/${required}" ] ||
        fail "the downloaded archive did not contain ${required}."
done

mkdir -p \
    "$BIN_DIR" \
    "$DATA_DIR/uploads" \
    "$DATA_DIR/translations" \
    "$DATA_DIR/logs" \
    "$RELEASES_DIR"

# Runtime state is shared by releases. A fresh source tree only contains
# .gitkeep files at these paths, so replacing those exact directories with
# symlinks cannot discard a user's data.
for directory in uploads translations logs; do
    if [ -e "${source_dir}/${directory}" ] || [ -L "${source_dir}/${directory}" ]; then
        rm -rf "${source_dir:?}/${directory}"
    fi
    ln -s "${DATA_DIR}/${directory}" "${source_dir}/${directory}"
done

for filename in translations.db cache.db; do
    if [ -e "${source_dir}/${filename}" ] || [ -L "${source_dir}/${filename}" ]; then
        rm -f "${source_dir}/${filename}"
    fi
    ln -s "${DATA_DIR}/${filename}" "${source_dir}/${filename}"
done

if [ -e "${source_dir}/ollama-server.log" ] || [ -L "${source_dir}/ollama-server.log" ]; then
    rm -f "${source_dir}/ollama-server.log"
fi
ln -s "${DATA_DIR}/logs/ollama-server.log" "${source_dir}/ollama-server.log"

release_id="$(date -u +%Y%m%d%H%M%S)-$$"
release_dir="${RELEASES_DIR}/${release_id}"
mv "$source_dir" "$release_dir"

for link in "${INSTALL_ROOT}/current" "${INSTALL_ROOT}/python"; do
    if [ -e "$link" ] && [ ! -L "$link" ]; then
        fail "${link} exists and is not an installer-managed symlink."
    fi
done
ln -sfn "$release_dir" "${INSTALL_ROOT}/current"
ln -sfn "$python_bin" "${INSTALL_ROOT}/python"

wrapper="${temporary_dir}/tolmach"
cat >"$wrapper" <<'EOF'
#!/bin/sh
# Installed by Tolmach install.sh.
set -eu
TOLMACH_HOME="${TOLMACH_HOME:-${HOME}/.local/share/tolmach}"
exec "${TOLMACH_HOME}/python" "${TOLMACH_HOME}/current/launch.py" "$@"
EOF
chmod 0755 "$wrapper"

installed_command="${BIN_DIR}/tolmach"
if [ -e "$installed_command" ] &&
    ! grep -Fq '# Installed by Tolmach install.sh.' "$installed_command"; then
    fail "${installed_command} already exists and was not created by this installer."
fi
mv "$wrapper" "$installed_command"

printf '\nTolmach installed.\n'
case ":${PATH}:" in
    *":${BIN_DIR}:"*) printf 'Run: tolmach\n' ;;
    *)
        printf 'Run: %s\n' "$installed_command"
        printf 'Add %s to PATH if you want to run it as tolmach.\n' "$BIN_DIR"
        ;;
esac
