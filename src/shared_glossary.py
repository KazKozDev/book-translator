"""Named glossaries that outlive one document.

The editable glossary is scoped to a single book, so a novel translated one
chapter at a time starts every chapter from an empty one and has the same
renderings corrected by hand again. A shared glossary is the memory between
those runs: a name plus a language pair, holding the entries the user approved.

Two rules keep it from turning into a liability:

- It only ever *overrides* what Stage 0 found in this text. Adding every stored
  term would put hundreds of names that never occur in the chapter into every
  prompt.
- It is written at Start, in the same transaction that creates the job — the
  moment the user approves the glossary — and never before.
"""

import sqlite3
from typing import Dict, Iterable, List, Optional, Tuple

from terminology import GlossaryTerm, TerminologyManager

MAX_NAME_LENGTH = 80

SCHEMA = '''
    CREATE TABLE IF NOT EXISTS shared_glossary_terms (
        name TEXT NOT NULL,
        source_lang TEXT NOT NULL,
        target_lang TEXT NOT NULL,
        -- The casefolded source term: the glossary parser already treats
        -- `Darcy` and `darcy` as one entry, so the store has to as well.
        source_key TEXT NOT NULL,
        source_term TEXT NOT NULL,
        target_term TEXT NOT NULL,
        enforcement_mode TEXT NOT NULL,
        note TEXT NOT NULL DEFAULT '',
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (name, source_lang, target_lang, source_key)
    );
'''

Scope = Tuple[str, str, str]


def scope(name, source_lang, target_lang) -> Scope:
    """Validate what identifies one shared glossary; raise ValueError if not."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Shared glossary name is required')
    name = name.strip()
    if len(name) > MAX_NAME_LENGTH:
        raise ValueError(
            f'Shared glossary name exceeds {MAX_NAME_LENGTH} characters'
        )
    if any(not char.isprintable() for char in name):
        raise ValueError('Shared glossary name contains control characters')
    if not isinstance(source_lang, str) or not source_lang.strip():
        raise ValueError('Source language is required')
    if not isinstance(target_lang, str) or not target_lang.strip():
        raise ValueError('Target language is required')
    return name, source_lang.strip(), target_lang.strip()


def load(conn: sqlite3.Connection, shared: Scope) -> List[GlossaryTerm]:
    rows = conn.execute(
        '''
        SELECT source_term, target_term, enforcement_mode, note
        FROM shared_glossary_terms
        WHERE name = ? AND source_lang = ? AND target_lang = ?
        ORDER BY rowid
        ''',
        shared,
    ).fetchall()
    return [GlossaryTerm(*row) for row in rows]


def store(
    conn: sqlite3.Connection,
    shared: Scope,
    terms: Iterable[GlossaryTerm],
    local_sources: Iterable[str] = (),
) -> int:
    """Write approved terms back; return how many were written.

    A term already stored is overwritten — the latest edit is the one the user
    just approved. Terms named in ``local_sources`` are specific to this text
    and are left out, so they do not leak into every later chapter.
    """
    local = {source.casefold() for source in local_sources}
    rows = [
        (*shared, term.source.casefold(), term.source, term.target, term.mode, term.note)
        for term in terms
        if term.source.casefold() not in local
    ]
    conn.executemany(
        '''
        INSERT INTO shared_glossary_terms (
            name, source_lang, target_lang, source_key,
            source_term, target_term, enforcement_mode, note, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(name, source_lang, target_lang, source_key)
        DO UPDATE SET
            source_term = excluded.source_term,
            target_term = excluded.target_term,
            enforcement_mode = excluded.enforcement_mode,
            note = excluded.note,
            updated_at = CURRENT_TIMESTAMP
        ''',
        rows,
    )
    return len(rows)


def listing(conn: sqlite3.Connection, source_lang: str, target_lang: str) -> List[Dict]:
    """Every shared glossary for one language pair, most recently used first."""
    rows = conn.execute(
        '''
        SELECT name, COUNT(*), MAX(updated_at)
        FROM shared_glossary_terms
        WHERE source_lang = ? AND target_lang = ?
        GROUP BY name
        ORDER BY MAX(updated_at) DESC, name
        ''',
        (source_lang, target_lang),
    ).fetchall()
    return [
        {'name': name, 'terms': count, 'updated_at': updated_at}
        for name, count, updated_at in rows
    ]


def delete(conn: sqlite3.Connection, shared: Scope) -> int:
    return conn.execute(
        '''
        DELETE FROM shared_glossary_terms
        WHERE name = ? AND source_lang = ? AND target_lang = ?
        ''',
        shared,
    ).rowcount


def to_text(terms: Iterable[GlossaryTerm]) -> str:
    return '\n'.join(
        TerminologyManager.format_line(term.source, term.target, term.mode, term.note)
        for term in terms
    )


def apply(
    records: List[Dict], shared_terms: Iterable[GlossaryTerm],
) -> Tuple[List[Dict], List[str]]:
    """Override the records Stage 0 proposed with their shared entries.

    Returns the records and the sources that were overridden. Only records
    Stage 0 found are touched; a shared term absent from this text is not
    added. Overridden records move to the end, in their original order, so the
    entries that still need reading come first and the ones settled in an
    earlier run sit together below them.
    """
    by_key: Dict[str, GlossaryTerm] = {
        term.source.casefold(): term for term in shared_terms
    }
    fresh, settled = [], []
    for record in records:
        term: Optional[GlossaryTerm] = by_key.get(record['source'].casefold())
        if term is None:
            fresh.append(record)
            continue
        settled.append({
            **record,
            'target': term.target,
            'mode': term.mode,
            'note': term.note,
        })
    return fresh + settled, [record['source'] for record in settled]
