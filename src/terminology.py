"""Per-book terminology constraints: the agreed rendering of every recurring
proper noun, and the checks that say whether the model honoured them.

Language-neutral by design — it knows nothing about which languages a run is
between, only about the terms it was given.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypedDict

import prompts


class ExactReplacement(TypedDict):
    """One enforced ``exact`` rule: the term, and how often it was rewritten."""

    source: str
    target: str
    count: int


@dataclass(frozen=True)
class GlossaryTerm:
    source: str
    target: str
    mode: str = "inflectable"
    #: Free text the author wrote about the term — its gender, its age, the
    #: ambiguity in the source that a rendering alone cannot settle. It reaches
    #: the models as prompt context and nothing else: unlike ``exact``, no check
    #: can enforce prose, so it is guidance the model may still ignore.
    note: str = ""


#: A trailing ``{note}`` on a glossary line. Braces rather than a third
#: delimiter because ``=>`` and ``|`` both occur inside terms, and a note is
#: free text that may contain either. Peeled from the end of the line before
#: the arrow is split, so nothing downstream has to know notes exist.
NOTE_SUFFIX = re.compile(r"\s*\{([^{}]*)\}\s*$")


def split_note(line: str) -> Tuple[str, str]:
    """Peel a trailing ``{note}`` off one line, without judging the rest.

    Deliberately tolerant: frontier verification exists to repair damaged
    entries, so it must be able to read a line ``from_text`` would reject.
    """
    match = NOTE_SUFFIX.search(line)
    if not match:
        return line.strip(), ""
    return line[:match.start()].strip(), match.group(1).strip()


class TerminologyManager:
    """Language-neutral, per-book terminology constraints."""

    VALID_MODES = {"exact", "inflectable", "preferred"}
    MAX_TERMS = 500
    MAX_TERM_LENGTH = 200
    MAX_NOTE_LENGTH = 200

    def __init__(self, terms: Optional[List[GlossaryTerm]] = None):
        deduplicated = {}
        for term in terms or []:
            deduplicated[term.source.casefold()] = term
        self.terms = list(deduplicated.values())

    @staticmethod
    def format_line(
        source: str, target: str, mode: str, note: str = "",
    ) -> str:
        """One glossary line in the textarea's format.

        Every place that rebuilds the text from stored terms goes through here,
        so a note written once survives reopening a job instead of being dropped
        the first time the editor is refilled from the database.
        """
        line = f"{source} => {target} | {mode}"
        return f"{line} {{{note}}}" if note else line

    @classmethod
    def from_text(cls, glossary_text: str):
        """Parse `source => target | mode {note}` or TSV lines; mode defaults to
        inflectable, note to nothing."""
        terms = []
        for line_number, raw_line in enumerate(glossary_text.splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            # Before anything else, because the note is the one field that can
            # hold text the arrow and the mode would otherwise fight over.
            line, note = split_note(line)
            if "{" in line or "}" in line:
                raise ValueError(
                    f"Glossary line {line_number}: an unbalanced brace — a note is "
                    f"written {{like this}} at the end of the line"
                )
            if note and len(note) > cls.MAX_NOTE_LENGTH:
                raise ValueError(
                    f"Glossary line {line_number}: a note exceeds "
                    f"{cls.MAX_NOTE_LENGTH} characters"
                )
            if line and not note and NOTE_SUFFIX.search(raw_line):
                # `Rom => Rom | inflectable {}`: the braces were there, so the
                # empty value is a mistake rather than an absent field.
                raise ValueError(f"Glossary line {line_number}: the note is empty")

            mode = "inflectable"
            if "\t" in line:
                parts = [part.strip() for part in line.split("\t")]
                if len(parts) not in (2, 3):
                    raise ValueError(
                        f"Glossary line {line_number}: use source<TAB>target<TAB>mode"
                    )
                source, target = parts[:2]
                if len(parts) == 3:
                    mode = parts[2].lower()
            else:
                separator = "=>" if "=>" in line else "=" if "=" in line else None
                if not separator:
                    raise ValueError(
                        f"Glossary line {line_number}: use source => target | mode"
                    )
                source, remainder = [part.strip() for part in line.split(separator, 1)]
                if "|" in remainder:
                    target, mode = [part.strip() for part in remainder.rsplit("|", 1)]
                    mode = mode.lower()
                else:
                    target = remainder.strip()

            if not source or not target:
                raise ValueError(f"Glossary line {line_number}: both terms are required")
            if len(source) > cls.MAX_TERM_LENGTH or len(target) > cls.MAX_TERM_LENGTH:
                raise ValueError(
                    f"Glossary line {line_number}: a term exceeds {cls.MAX_TERM_LENGTH} characters"
                )
            if mode not in cls.VALID_MODES:
                raise ValueError(
                    f"Glossary line {line_number}: mode must be exact, inflectable, or preferred"
                )
            terms.append(GlossaryTerm(
                source=source, target=target, mode=mode, note=note,
            ))

        if len(terms) > cls.MAX_TERMS:
            raise ValueError(f"Glossary supports at most {cls.MAX_TERMS} terms")
        return cls(terms)

    def relevant_terms(self, source_text: str) -> List[GlossaryTerm]:
        folded_text = source_text.casefold()
        return [term for term in self.terms if term.source.casefold() in folded_text]

    def prompt_context(self, source_text: str) -> str:
        relevant = self.relevant_terms(source_text)
        if not relevant:
            return ""

        lines = []
        any_note = False
        for term in relevant:
            rule = prompts.render("shared/terminology", f"mode_{term.mode}")
            if term.note:
                any_note = True
                lines.append(prompts.render(
                    "shared/terminology", "entry_with_note",
                    source=term.source, target=term.target,
                    rule=rule, note=term.note,
                ))
            else:
                lines.append(prompts.render(
                    "shared/terminology", "entry",
                    source=term.source, target=term.target, rule=rule,
                ))
        # A glossary with no notes anywhere renders exactly as it always has.
        # Only a glossary the author actually annotated pays for the extra
        # instruction explaining what a note is and that it is not text.
        block = "entries_with_notes" if any_note else prompts.MAIN
        # The two blank lines belong to the prompt this block is spliced into,
        # not to the block, so they are added here rather than in the file.
        return "\n\n" + prompts.render(
            "shared/terminology", block, entries="\n".join(lines),
        )

    def exact_violations(self, source_text: str, translated_text: str) -> List[Dict[str, str]]:
        translated_folded = translated_text.casefold()
        return [
            {"source": term.source, "required_target": term.target}
            for term in self.relevant_terms(source_text)
            if term.mode == "exact" and term.target.casefold() not in translated_folded
        ]

    def enforce_exact_source_forms(self, translated_text: str) -> Tuple[str, List[ExactReplacement]]:
        """Replace an exact term only when the model leaked its source form.

        A glossary is still provided to the model as translation context: it
        remains the only safe way to choose a rendering that is absent from
        the output.  But an ``exact`` rule has one deterministic case we can
        honour without guessing — the model translated the surrounding prose
        and left the literal source term unchanged.  Fix that case here, both
        for fresh generations and cached chunks.  ``inflectable`` and
        ``preferred`` terms are intentionally never rewritten this way.
        """
        replacements: List[ExactReplacement] = []
        result = translated_text
        for term in self.terms:
            if term.mode != "exact" or term.source.casefold() == term.target.casefold():
                continue
            # Do not turn a source substring inside a longer word into a
            # glossary term. ``\w`` is Unicode-aware, so this works for Latin,
            # Cyrillic and CJK source terms alike.
            pattern = re.compile(rf"(?<!\w){re.escape(term.source)}(?!\w)", re.IGNORECASE)
            result, count = pattern.subn(term.target, result)
            if count:
                replacements.append({
                    "source": term.source,
                    "target": term.target,
                    "count": count,
                })
        return result, replacements

    def fingerprint(self) -> str:
        # The note is part of the tuple on purpose. It changes the prompt the
        # models receive, so a glossary whose note was edited must not resolve
        # to the cached chunks translated under the old one.
        canonical = sorted(
            (term.source.casefold(), term.target, term.mode, term.note)
            for term in self.terms
        )
        payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
