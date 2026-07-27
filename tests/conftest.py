"""Keep the test run out of the application's own log directory.

``translator`` builds its file handlers at import time, so the redirect has to
be in place before the first test module imports it — which is what conftest is
for. Without this, every run of the suite writes its fixture warnings into
logs/translations.log, where they show up in the live log console beside real
chunks of a real book.
"""

import os
import tempfile

os.environ.setdefault(
    'TOLMACH_LOG_DIR', os.path.join(tempfile.gettempdir(), 'tolmach-test-logs'),
)
