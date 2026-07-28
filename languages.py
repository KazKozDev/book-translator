"""The languages the interface offers, by ISO code.

Read as ``LANG_NAMES.get(code, code)`` everywhere: a code with no entry is
passed to the model as-is rather than dropped, so adding a language to the
front end does not require a change here to keep working.
"""

LANG_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese',
    'ja': 'Japanese', 'ko': 'Korean'
}
