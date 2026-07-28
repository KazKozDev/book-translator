"""Per-book terminology constraints: the agreed rendering of every recurring
proper noun, and the checks that say whether the model honoured them.

Language-neutral by design — it knows nothing about which languages a run is
between, only about the terms it was given.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import prompts


@dataclass(frozen=True)
class GlossaryTerm:
    source: str
    target: str
    mode: str = "inflectable"


class TerminologyManager:
    """Language-neutral, per-book terminology constraints."""

    VALID_MODES = {"exact", "inflectable", "preferred"}
    MAX_TERMS = 500
    MAX_TERM_LENGTH = 200

    def __init__(self, terms: Optional[List[GlossaryTerm]] = None):
        deduplicated = {}
        for term in terms or []:
            deduplicated[term.source.casefold()] = term
        self.terms = list(deduplicated.values())

    @classmethod
    def from_text(cls, glossary_text: str):
        """Parse `source => target | mode` or TSV lines; mode defaults to inflectable."""
        terms = []
        for line_number, raw_line in enumerate(glossary_text.splitlines(), 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

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
            terms.append(GlossaryTerm(source=source, target=target, mode=mode))

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

        lines = [
            prompts.render(
                "shared/terminology", "entry",
                source=term.source,
                target=term.target,
                rule=prompts.render("shared/terminology", f"mode_{term.mode}"),
            )
            for term in relevant
        ]
        # The two blank lines belong to the prompt this block is spliced into,
        # not to the block, so they are added here rather than in the file.
        return "\n\n" + prompts.render(
            "shared/terminology", entries="\n".join(lines),
        )

    def exact_violations(self, source_text: str, translated_text: str) -> List[Dict[str, str]]:
        translated_folded = translated_text.casefold()
        return [
            {"source": term.source, "required_target": term.target}
            for term in self.relevant_terms(source_text)
            if term.mode == "exact" and term.target.casefold() not in translated_folded
        ]

    def enforce_exact_source_forms(self, translated_text: str) -> Tuple[str, List[Dict[str, str]]]:
        """Replace an exact term only when the model leaked its source form.

        A glossary is still provided to the model as translation context: it
        remains the only safe way to choose a rendering that is absent from
        the output.  But an ``exact`` rule has one deterministic case we can
        honour without guessing — the model translated the surrounding prose
        and left the literal source term unchanged.  Fix that case here, both
        for fresh generations and cached chunks.  ``inflectable`` and
        ``preferred`` terms are intentionally never rewritten this way.
        """
        replacements: List[Dict[str, str]] = []
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
        canonical = sorted(
            (term.source.casefold(), term.target, term.mode) for term in self.terms
        )
        payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
