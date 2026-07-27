#!/usr/bin/env python3
"""Cross-platform bootstrap for Book Translator.

It deliberately uses only Python's standard library until its virtual
environment is ready, so a freshly cloned checkout starts the same way on
macOS, Linux and Windows.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

# Standard library only, like everything else this file touches before the
# virtual environment exists — banner.py imports nothing else either.
from banner import print_terminal_banner


PROJECT_DIR = Path(__file__).resolve().parent
VENV_DIR = PROJECT_DIR / "venv"
IS_WINDOWS = platform.system() == "Windows"
PYTHON = VENV_DIR / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
OLLAMA_API = "http://127.0.0.1:11434/api/tags"
PORT = int(os.environ.get("PORT", "5001"))
# The smallest supported quality setup has two independent model roles:
# TranslateGemma for the first pass, and a general instruct model to reason
# about names, refine, verify edits, and judge.  Never use TranslateGemma as
# its own judge: it is translation-only and cannot provide an independent gate.
MINIMUM_TRANSLATION_MODEL = "translategemma:12b"
MINIMUM_JUDGE_MODEL = "qwen3:32b"


def log(mark: str, message: str) -> None:
    print(f"{mark} {message}", flush=True)


def fail(message: str) -> None:
    log("✗", message)
    if sys.stdin.isatty():
        input("Press Enter to close…")
    raise SystemExit(1)


def ask(question: str) -> bool:
    if not sys.stdin.isatty():
        return False
    return input(f"  {question} [y/N] ").strip().lower() in {"y", "yes", "д", "да"}


def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, **kwargs)


def ensure_venv() -> None:
    log("▸", "Virtual environment")
    if not PYTHON.exists():
        result = run([sys.executable, "-m", "venv", "--clear", str(VENV_DIR)])
        if result.returncode:
            fail("Could not create the virtual environment.")
        log("✓", "Created")
    else:
        log("✓", "Present")

    log("▸", "Dependencies")
    # This is a cheap no-op when pins are already met, and it guarantees a
    # cloned checkout sees changes in requirements.txt on its next launch.
    result = run([str(PYTHON), "-m", "pip", "install", "-r", "requirements.txt"], cwd=PROJECT_DIR)
    if result.returncode:
        fail("Could not install Python dependencies. See the pip output above.")
    log("✓", "Virtual environment activated for Book Translator")


def find_ollama() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    if IS_WINDOWS:
        candidate = Path(os.environ.get("LOCALAPPDATA", "")) / "Ollama" / "ollama.exe"
        if candidate.exists():
            return str(candidate)
    return None


def install_ollama() -> bool:
    system = platform.system()
    if system == "Darwin":
        brew = shutil.which("brew")
        if brew:
            return run([brew, "install", "--cask", "ollama"]).returncode == 0
        webbrowser.open("https://ollama.com/download/mac")
        log("!", "Homebrew is unavailable; the official Ollama installer was opened.")
        return False
    if system == "Windows":
        winget = shutil.which("winget")
        if winget:
            return run([winget, "install", "-e", "--id", "Ollama.Ollama", "--accept-package-agreements", "--accept-source-agreements"]).returncode == 0
        webbrowser.open("https://ollama.com/download/windows")
        log("!", "winget is unavailable; the official Ollama installer was opened.")
        return False
    if system == "Linux":
        curl = shutil.which("curl")
        if curl:
            # Ollama's published installer is the portable Linux installation
            # route. It runs only after the user answered this launcher's prompt.
            command = "curl -fsSL https://ollama.com/install.sh | sh"
            return run(["sh", "-c", command]).returncode == 0
        webbrowser.open("https://ollama.com/download/linux")
        log("!", "curl is unavailable; the official Ollama installation page was opened.")
        return False
    log("!", f"Automatic Ollama installation is not implemented for {system}.")
    return False


def ollama_up() -> bool:
    try:
        with urllib.request.urlopen(OLLAMA_API, timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def start_ollama(ollama: str) -> None:
    if ollama_up():
        log("✓", "Ollama server already running on port 11434")
        return
    if not ask("Ollama server is stopped. Start a separate server now?"):
        fail("Start Ollama with 'ollama serve', then launch Book Translator again.")
    logfile = (PROJECT_DIR / "ollama-server.log").open("a", encoding="utf-8")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0
    subprocess.Popen([ollama, "serve"], stdout=logfile, stderr=subprocess.STDOUT, cwd=PROJECT_DIR, creationflags=flags)
    for _ in range(30):
        if ollama_up():
            log("✓", "Ollama server is ready on port 11434")
            return
        time.sleep(0.5)
    fail(f"Ollama did not start. Read {PROJECT_DIR / 'ollama-server.log'} for details.")


def local_models(ollama: str) -> list[str]:
    listed = run([ollama, "list"], capture_output=True)
    if listed.returncode:
        return []
    models: list[str] = []
    for line in listed.stdout.splitlines()[1:]:
        columns = line.split()
        if columns and len(columns) >= 3 and columns[2] != "-":
            models.append(columns[0])
    return models


def ensure_pipeline_models(ollama: str) -> None:
    models = local_models(ollama)
    translation_model = next((model for model in models if "translategemma" in model.lower()), None)
    judge_model = next((model for model in models if "translategemma" not in model.lower()), None)
    missing: list[tuple[str, str]] = []
    if translation_model:
        log("✓", f"Translation model present: {translation_model}")
    else:
        missing.append(("translation", MINIMUM_TRANSLATION_MODEL))
    if judge_model:
        log("✓", f"Independent judge model present: {judge_model}")
    else:
        missing.append(("independent judge", MINIMUM_JUDGE_MODEL))

    if not missing:
        log("✓", f"{len(models)} local model(s) available")
        return

    labels = ", ".join(f"{role}: {model}" for role, model in missing)
    log("!", f"Missing required model role(s): {labels}")
    log("·", "The minimum pipeline is TranslateGemma + a separate instruct judge.")
    if not ask("Download the missing model(s) now? These are large downloads"):
        fail("The pipeline requires a TranslateGemma translation model and a separate local judge model.")
    for role, model in missing:
        log("·", f"Downloading {role} model: {model}")
        result = run([ollama, "pull", model], cwd=PROJECT_DIR)
        if result.returncode:
            fail(f"Could not download {model}. Check network and disk space.")
        log("✓", f"Ready: {model}")


def listening_pids() -> list[int]:
    # psutil is deliberately executed inside the freshly prepared venv; the
    # system Python running this bootstrap need not have project packages.
    code = (
        "import json, psutil; "
        "print(json.dumps(sorted({c.pid for c in psutil.net_connections(kind='inet') "
        "if c.pid and c.status == 'LISTEN' and c.laddr.port == " + str(PORT) + "})))"
    )
    result = run([str(PYTHON), "-c", code], capture_output=True)
    try:
        return [int(pid) for pid in json.loads(result.stdout)]
    except (json.JSONDecodeError, ValueError):
        return []


def free_port() -> None:
    pids = listening_pids()
    if not pids:
        return
    log("!", f"Port {PORT} is in use by PID(s): {', '.join(map(str, pids))}; restarting the app")
    for pid in pids:
        if IS_WINDOWS:
            run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    time.sleep(1)


def port_ready() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=1) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def start_translator() -> None:
    free_port()
    log("▸", "Book Translator")
    env = os.environ.copy()
    # This launcher prints the banner itself, at the top where it belongs, so
    # the server is told not to print it a second time.
    env["TOLMACH_BANNER_PRINTED"] = "1"
    server = subprocess.Popen([str(PYTHON), "translator.py"], cwd=PROJECT_DIR, env=env)
    for _ in range(30):
        if port_ready():
            url = f"http://127.0.0.1:{PORT}/?launched={int(time.time())}"
            log("✓", f"Ready: {url}")
            webbrowser.open(url)
            try:
                server.wait()
            except KeyboardInterrupt:
                server.terminate()
            return
        if server.poll() is not None:
            fail("translator.py exited before opening the interface.")
        time.sleep(0.5)
    server.terminate()
    fail(f"translator.py did not open port {PORT}.")


def main() -> None:
    if sys.version_info < (3, 9):
        fail("Python 3.9 or later is required.")
    os.chdir(PROJECT_DIR)
    # The logo first, then where and when this is running, then the checks —
    # the order the shell launcher had before this file replaced it.
    print_terminal_banner(force=True)
    log(" ", f"{time.strftime('%a %d %b %Y, %H:%M')} · {platform.system()} {platform.release()}")
    log(" ", str(PROJECT_DIR))
    print()
    log("▸", f"Book Translator setup ({platform.system()})")
    ensure_venv()
    log("▸", "Ollama")
    ollama = find_ollama()
    if not ollama:
        if not ask("Ollama is not installed. Install it now?"):
            fail("Ollama is required for translation.")
        if not install_ollama():
            fail("Complete the Ollama installation, then run this launcher again.")
        ollama = find_ollama()
        if not ollama:
            fail("Ollama was installed but is not yet on PATH. Close and reopen the launcher.")
    log("✓", f"Ollama: {ollama}")
    start_ollama(ollama)
    ensure_pipeline_models(ollama)
    start_translator()


if __name__ == "__main__":
    main()
