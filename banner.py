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
    padding = ' ' * max(0, (61 - len(SUBTITLE)) // 2)
    print(f'\n\n\n{accent}{TERMINAL_LOGO}{reset}')
    print(f'\n{cream}{padding}{SUBTITLE}{reset}\n\n')
