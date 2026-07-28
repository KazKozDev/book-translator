"""The words the models are sent, loaded from ``prompts/``.

No prompt text lives in the Python files any more. Each role has its own file
under ``prompts/``, so the whole instruction a model receives can be read, and
edited, without going through the pipeline that assembles it.

The format is deliberately small:

* ``$name`` is a placeholder, filled by the caller. Anything else — including
  the ``{`` and ``}`` of the JSON shapes the prompts ask for — is literal, so
  the file reads exactly like what the model will be sent.
* A line beginning with ``## `` starts a named section. Text before the first
  such line is the main section, which is what ``render`` returns by default.
  Sections hold the optional blocks a prompt grows when there is context to
  add: the previous paragraph, a glossary, a list of violated terms.
* Blank lines around a section heading are markup, not content — every
  section is stripped of leading and trailing blank lines. Where a block has
  to be joined to its prompt by a specific number of newlines, the joining is
  done at the call site, because that is layout rather than wording.

A missing placeholder value raises rather than rendering an empty string: a
prompt that silently loses its source text is worse than one that fails.
"""

from pathlib import Path
from string import Template
from typing import Dict, List, Set

PROMPTS_DIR = Path(__file__).resolve().parent / 'prompts'

#: The section returned when none is named — the prompt itself.
MAIN = ''

_cache: Dict[str, Dict[str, str]] = {}


def _split_sections(text: str) -> Dict[str, str]:
    sections: Dict[str, List[str]] = {MAIN: []}
    current = MAIN
    for line in text.split('\n'):
        if line.startswith('## '):
            current = line[3:].strip()
            sections[current] = []
            continue
        sections[current].append(line)
    return {name: '\n'.join(lines).strip('\n') for name, lines in sections.items()}


def _path(name: str) -> Path:
    return PROMPTS_DIR.joinpath(*name.split('/')).with_suffix('.md')


def load(name: str) -> Dict[str, str]:
    """Every section of one prompt file, read once and kept."""
    if name not in _cache:
        path = _path(name)
        if not path.is_file():
            raise FileNotFoundError(f"No prompt file at {path}")
        _cache[name] = _split_sections(path.read_text(encoding='utf-8'))
    return _cache[name]


def render(name: str, section: str = MAIN, **values: object) -> str:
    """One prompt, or one of its optional blocks, with its values filled in."""
    sections = load(name)
    if section not in sections:
        raise KeyError(
            f"prompts/{name}.md has no section '{section}' — "
            f"it has {sorted(key for key in sections if key) or 'no sections'}"
        )
    return Template(sections[section]).substitute(values)


def names() -> List[str]:
    """Every prompt file that exists, by the name ``render`` takes."""
    return sorted(
        path.relative_to(PROMPTS_DIR).with_suffix('').as_posix()
        for path in PROMPTS_DIR.rglob('*.md')
        if path.name != 'README.md'
    )


def loaded() -> Set[str]:
    """The prompt files read so far this process. Used by the tests to catch a
    file that no code path can reach any more."""
    return set(_cache)
