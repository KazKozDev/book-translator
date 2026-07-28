"""The startup banner, in one place and with no dependencies.

Both entry points show it: ``launch.py`` before it starts checking the machine,
and ``translator.py`` when it is run directly. It lives in its own stdlib-only
module because the launcher runs before the virtual environment exists and so
cannot import ``translator`` (which needs Flask) — and because a second copy of
the artwork is exactly how the logo goes missing from one of the two paths.
"""

import os
import sys

TERMINAL_LOGO = """████████╗ ██████╗ ██╗     ███╗   ███╗ █████╗  ██████╗██╗  ██╗
╚══██╔══╝██╔═══██╗██║     ████╗ ████║██╔══██╗██╔════╝██║  ██║
   ██║   ██║   ██║██║     ██╔████╔██║███████║██║     ███████║
   ██║   ██║   ██║██║     ██║╚██╔╝██║██╔══██║██║     ██╔══██║
   ██║   ╚██████╔╝███████╗██║ ╚═╝ ██║██║  ██║╚██████╗██║  ██║
   ╚═╝    ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝"""

# The same logo in characters cp1252 can represent. A Windows console that has
# not been switched to UTF-8 encodes stdout as cp1252, and printing the block
# drawing above raises UnicodeEncodeError — which, in launch.py, happens before
# anything useful has been said and looks like the app failing to start.
ASCII_LOGO = r"""  _____ ___  _     __  __    _    ____ _   _
 |_   _/ _ \| |   |  \/  |  / \  / ___| | | |
   | || | | | |   | |\/| | / _ \| |   | |_| |
   | || |_| | |___| |  | |/ ___ \ |___|  _  |
   |_| \___/|_____|_|  |_/_/   \_\____|_| |_|"""

SUBTITLE = 'B O O K   T R A N S L A T O R  v3.0'
# Tones taken from :root in static/index.html.
_ACCENT = '\033[1m\033[38;2;91;124;153m'
_CREAM = '\033[1m\033[38;2;240;214;170m'
_RESET = '\033[0m'


def print_terminal_banner(force: bool = False) -> None:
    """Print the logo, unless a launcher has already printed it.

    ``TOLMACH_BANNER_PRINTED`` is how the launcher tells the server it has the
    banner covered, so a launched run does not show it twice. ``force`` is for
    the launcher itself, which prints first and sets the variable afterwards.
    """
    if not force and os.environ.get('TOLMACH_BANNER_PRINTED'):
        return

    use_color = sys.stdout.isatty() and not os.environ.get('NO_COLOR')
    accent = _ACCENT if use_color else ''
    cream = _CREAM if use_color else ''
    reset = _RESET if use_color else ''
    logo = TERMINAL_LOGO if _printable(TERMINAL_LOGO) else ASCII_LOGO
    padding = ' ' * max(0, (61 - len(SUBTITLE)) // 2)
    print(f'\n\n\n{accent}{logo}{reset}')
    print(f'\n{cream}{padding}{SUBTITLE}{reset}\n\n')


def _printable(text: str) -> bool:
    """Whether stdout's encoding can carry ``text``.

    Asked rather than caught: a half-written banner is worse than a plain one,
    and print() raises after emitting the lines that did encode.
    """
    encoding = getattr(sys.stdout, 'encoding', None) or 'ascii'
    try:
        text.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True
