"""Stage 3: the standalone quality tests behind the Tests panel.

One class split across two files rather than a reusable behaviour — these are
BookTranslator methods, and they are here because a thousand lines of optional
diagnostics were burying the pipeline they diagnose. Mixed in, not imported, so
every call site reads exactly as it did before.

What it expects the host class to provide: ``model_name``, ``session``,
``api_url``, ``_WORD_RE``, ``_ollama_payload``, ``_ollama_response_text``,
``stage1_primary_translation`` and ``harvest_proper_noun_candidates``.
"""

import difflib
import os
import random
import re
import threading
import unicodedata
from collections import Counter
from statistics import mean, median
from typing import Dict, List, Optional, Tuple

try:
    import sacrebleu
except ImportError:
    sacrebleu = None

import prompts
from languages import LANG_NAMES
from monitoring import logger
from terminology import TerminologyManager


class QualityTests:
    # Optional document diagnostics. They stay lazy because loading either
    # model is a deliberate quality-check action, never a prerequisite for
    # translating a book.
    LABSE_MODEL_ID = 'sentence-transformers/LaBSE'
    LANGUAGE_ID_MODEL_ID = 'papluca/xlm-roberta-base-language-detection'
    _labse_model = None
    _labse_lock = threading.Lock()
    _language_id_pipeline = None
    _language_id_lock = threading.Lock()
    # ------------------------------------------------------------------
    # Stage 3: standalone quality tests, run on demand from the UI after
    # Stage 1 and/or Stage 2 — never automatically. Several of these call
    # the model again (backtranslation, LLM-judge) or a QE model
    # (COMET-Kiwi), which is too slow to run on every chunk of a full
    # book, so they sample a handful of chunks instead of the whole text.
    # ------------------------------------------------------------------

    EVAL_SAMPLE_SIZE = 5

    @staticmethod
    def _sample_indices(length: int, sample_size: int) -> List[int]:
        """Evenly spaced indices across [0, length), for sampling chunks
        without re-processing an entire book on every test run."""
        if length <= 0:
            return []
        k = min(sample_size, length)
        if k <= 1:
            return [0]
        return sorted({round(i * (length - 1) / (k - 1)) for i in range(k)})

    # Dialogue openers across the conventions this app's language list uses.
    _DIALOGUE_RE = re.compile(r'["“”«»„]|(?:^|\n)\s*[—–-]\s')

    @classmethod
    def _risk_ranked_indices(cls, chunks: List[str], sample_size: int) -> List[int]:
        """Indices of the riskiest chunks, rather than evenly spaced ones.

        Evenly spaced sampling weights every chunk equally, but translation
        errors are not evenly spaced: they cluster where there are names to
        render consistently, dialogue to keep in register, and numbers to
        carry over exactly. Spending the same five model calls on those
        chunks finds strictly more than spending them on scenery.

        Falls back to even spacing when nothing scores — a chunk list with no
        names, no dialogue and no numbers has no risk profile to rank by.
        """
        scored = []
        for index, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            names = len(cls.harvest_proper_noun_candidates(chunk, limit=20))
            dialogue = len(cls._DIALOGUE_RE.findall(chunk))
            digits = sum(character.isdigit() for character in chunk)
            score = names * 3 + min(dialogue, 10) + min(digits, 10)
            if score:
                scored.append((score, index))

        if not scored:
            return cls._sample_indices(len(chunks), sample_size)
        # Highest score first, index as the tie-break so a rerun samples the
        # same chunks and its numbers stay comparable to the previous run's.
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return sorted(index for _, index in scored[:sample_size])


    def eval_length_ratio(self, source_text: str, draft_text: str, final_text: str) -> Dict:
        """Final length against the SOURCE, with final-against-draft as a
        secondary number.

        Measuring the final against the draft answers a question nobody is
        asking: both were produced by the same pipeline from the same text, so
        the ratio sits near 1.00 whatever happened to the meaning. Measuring
        against the source is what shows compression — a target text that
        comes out the same length as an English source is suspicious, because
        most target languages expand.
        """
        source_len, draft_len, final_len = len(source_text), len(draft_text), len(final_text)
        ratio = (final_len / source_len) if source_len else 1.0
        draft_ratio = (final_len / draft_len) if draft_len else 1.0
        # A deliberately wide band: expansion factors are language-pair
        # specific (EN→RU runs 1.10–1.20, EN→ZH well under 1), so this can
        # only catch the gross cases — a target half the size of its source,
        # or twice it — without a per-pair table to compare against.
        flagged = not (0.55 <= ratio <= 1.6)
        return {
            'test': 'length_ratio',
            'label': 'Length ratio (final / source)',
            'value': round(ratio, 3),
            'details': {
                'source_chars': source_len,
                'draft_chars': draft_len,
                'final_chars': final_len,
                'final_over_draft': round(draft_ratio, 3),
            },
            'flagged': flagged,
            'note': (
                f"Final is {ratio:.2f}x the source's length ({source_len} → {final_len} chars) "
                f"— outside 0.55–1.6x, so text was probably lost or duplicated. "
                f"Final/draft {draft_ratio:.2f}x."
                if flagged else
                f"Final is {ratio:.2f}x the source's length ({source_len} → {final_len} chars); "
                f"final/draft {draft_ratio:.2f}x. Compare against what your language pair "
                f"normally does — a ratio near 1.00 into a language that usually expands "
                f"means the translation is compressing."
            ),
        }

    def eval_diff_ratio(self, draft_text: str, final_text: str) -> Dict:
        # autojunk MUST stay off. It treats any character occurring in more
        # than 1% of a long sequence as junk to be ignored — which in prose is
        # most of the alphabet, so it reports two nearly identical texts as
        # wildly different. On a measured example the same pair scored 0.82
        # with autojunk and 0.98 without; 0.98 was the truth.
        ratio = difflib.SequenceMatcher(None, draft_text, final_text, autojunk=False).ratio()
        # Refinement is now a span patcher, not a rewriter, so a high
        # similarity is the expected outcome and no longer a complaint. A LOW
        # one is the alarm: it means something rewrote the text wholesale.
        flagged = ratio < 0.75
        if ratio < 0.75:
            note = f"Similarity {ratio:.2f} — over a quarter of the text changed. Refinement patches reported spans, so this much movement means something rewrote the text wholesale; check for hallucination or duplication."
        elif ratio > 0.995:
            note = f"Similarity {ratio:.2f} — refinement changed almost nothing. Either the draft was already clean or the review pass found nothing it could locate; check how many errors it reported."
        else:
            note = f"Similarity {ratio:.2f} — {(1 - ratio) * 100:.1f}% of the text changed, which is the range a span-level patch should land in."
        return {
            'test': 'diff_ratio',
            'label': 'Draft/final similarity',
            'value': round(ratio, 3),
            'flagged': flagged,
            'note': note,
        }

    def eval_ngram_repetition(self, text: str, n: int = 4) -> Dict:
        words = text.split()
        if len(words) < n * 2:
            return {
                'test': 'ngram_repetition',
                'label': 'Repeated phrase ratio',
                'value': 0.0,
                'flagged': False,
                'note': 'Text too short to evaluate.',
            }
        ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        repeated = sum(count for count in counts.values() if count > 1)
        ratio = repeated / len(ngrams)
        flagged = ratio > 0.15
        return {
            'test': 'ngram_repetition',
            'label': 'Repeated phrase ratio',
            'value': round(ratio, 3),
            'flagged': flagged,
            'note': (
                f"{ratio:.0%} of {n}-word phrases repeat — likely duplicated passages."
                if flagged else
                f"{ratio:.0%} of {n}-word phrases repeat — normal for natural prose."
            ),
        }

    def eval_terminology_delta(
        self, original_text: str, draft_text: str, final_text: str,
        terminology: 'TerminologyManager',
    ) -> Dict:
        draft_violations = terminology.exact_violations(original_text, draft_text)
        final_violations = terminology.exact_violations(original_text, final_text)
        delta = len(draft_violations) - len(final_violations)
        if len(draft_violations) == 0 and len(final_violations) == 0:
            note = 'No verified terms were violated in either pass.'
        elif delta > 0:
            note = f'Refinement fixed {delta} glossary violation(s) ({len(draft_violations)} → {len(final_violations)}).'
        elif delta < 0:
            note = f'Refinement introduced {-delta} new glossary violation(s) ({len(draft_violations)} → {len(final_violations)}).'
        else:
            note = f'Refinement left {len(final_violations)} glossary violation(s) unfixed.'
        return {
            'test': 'terminology_delta',
            'label': 'Glossary violations (draft vs final)',
            'value': delta,
            'details': {
                'draft_violations': len(draft_violations),
                'final_violations': len(final_violations),
            },
            'flagged': len(final_violations) > 0,
            'note': note,
        }

    # -- Deterministic document-level checks. No model, no sampling: these
    # read the whole final text, which is what the LLM-judge tests can never
    # do, and they are the only tests here whose answer is a fact rather
    # than an opinion. --

    @staticmethod
    def _script_of(character: str) -> Optional[str]:
        """The script a letter belongs to ('LATIN', 'CYRILLIC', 'CJK', …),
        or None for anything that isn't a letter.

        Read out of the Unicode character name rather than a table of code
        point ranges, so it covers every script without this file having to
        know which languages exist."""
        if not character.isalpha():
            return None
        try:
            return unicodedata.name(character).split()[0]
        except ValueError:
            return None

    @classmethod
    def dominant_script(cls, text: str) -> Optional[str]:
        """The script most of a text's letters are written in."""
        counts = Counter(
            script for script in (cls._script_of(character) for character in text) if script
        )
        return counts.most_common(1)[0][0] if counts else None

    def eval_script_leakage(self, source_text: str, final_text: str) -> Dict:
        """Words in the final translation still written in the source's
        script.

        This is the cheapest possible check for a name that was never
        translated at all — "Grunnings" sitting in the middle of a Cyrillic
        page — and no LLM judge is needed to see it. Some leakage is
        legitimate (a brand kept in Latin on purpose), so the words are
        listed rather than just counted, and the reader decides.
        """
        target_script = self.dominant_script(final_text)
        source_script = self.dominant_script(source_text)
        if not target_script or not source_script or target_script == source_script:
            return {
                'test': 'script_leakage',
                'label': 'Untranslated source-script words',
                'value': 0,
                'flagged': False,
                'note': (
                    'Source and target use the same script, so untranslated words '
                    'cannot be told apart this way.'
                    if target_script and source_script else
                    'Not enough text to determine the scripts involved.'
                ),
            }

        leaked = Counter()
        for match in self._WORD_RE.finditer(final_text):
            word = match.group(0)
            if self.dominant_script(word) == source_script:
                leaked[word] += 1

        total = sum(leaked.values())
        examples = ', '.join(
            f'{word} ({count}x)' if count > 1 else word
            for word, count in leaked.most_common(8)
        )
        return {
            'test': 'script_leakage',
            'label': 'Untranslated source-script words',
            'value': total,
            'details': {
                'distinct_words': len(leaked),
                'target_script': target_script.title(),
                'source_script': source_script.title(),
                'words': dict(leaked.most_common(20)),
            },
            'flagged': total > 0,
            'note': (
                f'{total} {source_script.title()}-script word(s) left in the '
                f'{target_script.title()} translation, {len(leaked)} distinct: {examples}. '
                'Check each one — a name left in the source script is a missed '
                'translation, a brand kept on purpose is not.'
                if total else
                f'No {source_script.title()}-script words left in the '
                f'{target_script.title()} translation.'
            ),
        }

    # How much of a target term has to match for two surface forms to count
    # as the same name inflected, rather than two different renderings. Names
    # inflect at the end, so the comparison is on the leading characters.
    ENTITY_STEM_MIN = 4

    @classmethod
    def _entity_stem(cls, term: str) -> str:
        """The leading part of a target term that inflection leaves alone."""
        head = term.split()[-1] if term.split() else term
        if len(head) <= cls.ENTITY_STEM_MIN:
            return head.casefold()
        return head[:max(cls.ENTITY_STEM_MIN, len(head) - 2)].casefold()

    @classmethod
    def _matches_entity_rendering(cls, candidate: str, target: str) -> bool:
        """Whether one target-language word can be an inflected rendering.

        A stem alone allows case endings, but must never let a shorter word
        stand in for the agreed name: ``Фенвик`` is not an inflection of
        ``Фенвикс``. This is deliberately conservative rather than pretending
        to provide morphology for every supported target language.
        """
        head = target.split()[-1] if target.split() else target
        folded = candidate.casefold()
        return len(folded) >= len(head) and folded.startswith(cls._entity_stem(target))

    def eval_entity_consistency(
        self, original_chunks: List[str], final_chunks: List[str],
        terminology: 'TerminologyManager',
    ) -> Dict:
        """Is every agreed rendering actually used everywhere its name occurs?

        Runs per chunk over the whole document, which is the point: a chunk
        can be internally perfect and still call the family something
        different from what the previous chunk called it. Any inflected form
        counts as a use — matching is on the stem, since a name that cannot
        take target-language endings would make the sentence around it
        ungrammatical.

        Reports the chunks where a name's source appears but no form of its
        agreed rendering does. That is the signature of a name being dropped,
        translated by meaning in one place and transcribed in another, or
        left in the source script.
        """
        if not terminology.terms:
            return {
                'test': 'entity_consistency',
                'label': 'Named-entity consistency',
                'value': None,
                'flagged': True,
                'note': (
                    'No terms to check. Run Prepare before Start (or fill the glossary '
                    'by hand) — with an empty glossary this test can only ever pass, '
                    'which for a text whose main risk is proper nouns is the most '
                    'expensive check to skip.'
                ),
            }

        # Two names whose stems are prefixes of each other cannot be told
        # apart by stem matching: "Дурсль" and "Дурсли" both reduce to
        # "Дурс", so a chunk that says the singular where the source has the
        # plural still counts as satisfied. Stem matching is what makes
        # inflection acceptable, and without morphology for the target
        # language the two requirements genuinely conflict — so the pairs are
        # named as needing a human eye rather than quietly passed.
        stems = {term.source: self._entity_stem(term.target) for term in terminology.terms}
        ambiguous = sorted({
            tuple(sorted((left, right)))
            for left, left_stem in stems.items()
            for right, right_stem in stems.items()
            if left != right and (left_stem.startswith(right_stem) or right_stem.startswith(left_stem))
        })

        pairs = list(zip(original_chunks, final_chunks))
        findings, checked = [], 0
        for term in terminology.terms:
            folded_source = term.source.casefold()
            occurrences, satisfied, forms = 0, 0, Counter()
            for original_chunk, final_chunk in pairs:
                if folded_source not in original_chunk.casefold():
                    continue
                occurrences += 1
                chunk_forms = [
                    match.group(0) for match in self._WORD_RE.finditer(final_chunk)
                    if self._matches_entity_rendering(match.group(0), term.target)
                ]
                if chunk_forms:
                    satisfied += 1
                    forms.update(chunk_forms)
            if not occurrences:
                continue
            checked += 1
            if satisfied < occurrences:
                findings.append({
                    'source': term.source,
                    'target': term.target,
                    'chunks_with_source': occurrences,
                    'chunks_with_rendering': satisfied,
                    'forms_used': [form for form, _ in forms.most_common(8)],
                })

        findings.sort(key=lambda finding: finding['chunks_with_source'] - finding['chunks_with_rendering'], reverse=True)
        if not checked:
            note = 'None of the glossary terms occur in the source text, so nothing was checked.'
        elif not findings:
            note = f'All {checked} term(s) that occur in the source are rendered in every chunk they appear in.'
        else:
            worst = '; '.join(
                f'"{finding["source"]}" → "{finding["target"]}" missing from '
                f'{finding["chunks_with_source"] - finding["chunks_with_rendering"]} of '
                f'{finding["chunks_with_source"]} chunk(s)'
                for finding in findings[:4]
            )
            note = f'{len(findings)} of {checked} term(s) are not rendered everywhere: {worst}.'

        if ambiguous:
            listed = '; '.join(f'{left} / {right}' for left, right in ambiguous[:4])
            note += (
                f' Cannot distinguish these renderings automatically, so check them by '
                f'hand: {listed}. They differ only in the endings that inflection is '
                f'allowed to change.'
            )

        return {
            'test': 'entity_consistency',
            'label': 'Named-entity consistency',
            'value': len(findings),
            'details': {
                'terms_checked': checked,
                'findings': findings[:20],
                'ambiguous_pairs': [list(pair) for pair in ambiguous[:20]],
            },
            'flagged': bool(findings),
            'note': note,
        }

    _NUMBER_RE = re.compile(
        r'(?<![\w.,])(?:\d{1,3}(?:[\s,\u00a0]\d{3})+|\d+)(?:[.,]\d+)?(?![\w.,])'
    )

    @classmethod
    def _numeric_tokens(cls, text: str) -> Counter:
        """Numbers in a comparison-safe spelling.

        This deliberately does not guess that ``one`` and ``один`` are the
        same number. Digit-bearing facts (years, dates, prices, quantities,
        section numbers) are factual and can be checked across almost every
        language pair; spelled-out numbers need language-specific parsing.
        """
        values = []
        for match in cls._NUMBER_RE.finditer(text):
            token = match.group(0).replace('\u00a0', '').replace(' ', '')
            # 1,000 is a thousands separator; 1,5 is a decimal comma. The
            # former has exactly three digits after its last separator.
            if ',' in token and token.rsplit(',', 1)[1].isdigit() and len(token.rsplit(',', 1)[1]) == 3:
                token = token.replace(',', '')
            elif ',' in token:
                token = token.replace(',', '.')
            values.append(token)
        return Counter(values)

    def eval_numeric_preservation(self, source_text: str, final_text: str) -> Dict:
        """Deterministic gate for digit-written numbers and dates."""
        source_values = self._numeric_tokens(source_text)
        final_values = self._numeric_tokens(final_text)
        missing = list((source_values - final_values).elements())
        unexpected = list((final_values - source_values).elements())
        if not source_values:
            return {
                'test': 'numeric_preservation',
                'label': 'Numbers and dates',
                'value': 0,
                'flagged': False,
                'details': {'source_values': [], 'missing': [], 'unexpected': []},
                'note': 'No digit-written numbers or dates in the source to check.',
            }
        flagged = bool(missing)
        missing_preview = ', '.join(missing[:12])
        unexpected_preview = ', '.join(unexpected[:12])
        note = (
            f'Missing or changed source value(s): {missing_preview}. '
            'Numbers and dates are facts; inspect the corresponding chunks before shipping.'
            if missing else
            f'All {sum(source_values.values())} digit-written source number(s)/date component(s) survive in the final.'
        )
        if unexpected:
            note += f' Extra target value(s) to inspect: {unexpected_preview}.'
        return {
            'test': 'numeric_preservation',
            'label': 'Numbers and dates',
            'value': len(missing),
            'flagged': flagged,
            'details': {
                'source_values': sorted(source_values.elements()),
                'missing': missing[:50],
                'unexpected': unexpected[:50],
            },
            'note': note,
        }

    def eval_chunk_coverage(self, original_chunks: List[str], final_chunks: List[str]) -> Dict:
        """Detect a missing or blank translation segment without a model."""
        source_count, final_count = len(original_chunks), len(final_chunks)
        empty_final = [
            index + 1 for index, source in enumerate(original_chunks)
            if source.strip() and (index >= final_count or not final_chunks[index].strip())
        ]
        count_mismatch = source_count != final_count
        flagged = count_mismatch or bool(empty_final)
        if flagged:
            parts = []
            if count_mismatch:
                parts.append(f'{source_count} source chunk(s), {final_count} final chunk(s)')
            if empty_final:
                parts.append('empty final chunk(s): ' + ', '.join(map(str, empty_final[:20])))
            note = 'Chunk coverage failed — ' + '; '.join(parts) + '.'
        else:
            note = f'All {source_count} source chunks have a non-empty aligned final chunk.'
        return {
            'test': 'chunk_coverage',
            'label': 'Chunk coverage',
            'value': len(empty_final) + abs(source_count - final_count),
            'flagged': flagged,
            'details': {
                'source_chunks': source_count,
                'final_chunks': final_count,
                'empty_final_chunks': empty_final[:50],
            },
            'note': note,
        }

    @classmethod
    def _get_labse_model(cls):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                'sentence-transformers is missing from the environment — the venv is '
                'incomplete. Run: ./venv/bin/python -m pip install -r requirements.txt'
            ) from exc
        if cls._labse_model is None:
            with cls._labse_lock:
                if cls._labse_model is None:
                    cls._labse_model = SentenceTransformer(cls.LABSE_MODEL_ID)
        return cls._labse_model

    def eval_labse_alignment(self, original_chunks: List[str], final_chunks: List[str]) -> Dict:
        """Document-wide source/final alignment and semantic-drift outliers.

        Chunks are already source-aligned by the translation pipeline. LaBSE
        measures every pair in one shared multilingual embedding space; it is
        a drift signal, not a fabricated claim that it can prove correctness.
        """
        length = min(len(original_chunks), len(final_chunks))
        if not length:
            return {
                'test': 'labse_alignment', 'label': 'LaBSE document alignment',
                'value': None, 'flagged': True,
                'note': 'No aligned source/final chunks available for LaBSE.',
            }
        model = self._get_labse_model()
        source = original_chunks[:length]
        final = final_chunks[:length]
        embeddings = model.encode(source + final, normalize_embeddings=True, show_progress_bar=False)
        scores = [
            float(sum(left * right for left, right in zip(embeddings[index], embeddings[length + index])))
            for index in range(length)
        ]
        baseline = median(scores)
        # A document-relative threshold catches the one paragraph that lost
        # meaning without imposing an English-centric absolute score.
        threshold = max(0.20, baseline - 0.20)
        flags = [
            {'chunk': index + 1, 'similarity': round(score, 3)}
            for index, score in enumerate(scores) if score < threshold
        ]
        return {
            'test': 'labse_alignment',
            'label': 'LaBSE document alignment',
            'value': round(mean(scores), 3),
            'flagged': bool(flags),
            'details': {
                'chunks_compared': length,
                'median_similarity': round(baseline, 3),
                'drift_threshold': round(threshold, 3),
                'drift_flags': flags[:50],
                'lowest_chunks': [
                    {'chunk': index + 1, 'similarity': round(score, 3)}
                    for index, score in sorted(enumerate(scores), key=lambda pair: pair[1])[:10]
                ],
            },
            'note': (
                f'{len(flags)} semantic-drift outlier(s) below the document-relative '
                f'threshold {threshold:.2f}; inspect those source/final chunks.'
                if flags else
                f'{length} aligned chunk pair(s), mean similarity {mean(scores):.2f}; '
                'no document-relative drift outlier.'
            ),
        }

    @classmethod
    def _get_language_id_pipeline(cls):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                'transformers is missing from the environment — the venv is incomplete. '
                'Run: ./venv/bin/python -m pip install -r requirements.txt'
            ) from exc
        if cls._language_id_pipeline is None:
            with cls._language_id_lock:
                if cls._language_id_pipeline is None:
                    cls._language_id_pipeline = pipeline(
                        'text-classification', model=cls.LANGUAGE_ID_MODEL_ID, tokenizer=cls.LANGUAGE_ID_MODEL_ID,
                    )
        return cls._language_id_pipeline

    def eval_language_id(self, final_chunks: List[str], target_lang: str) -> Dict:
        """Flag non-target-language final segments, including untranslated text."""
        expected = (target_lang or '').casefold()
        classifier = self._get_language_id_pipeline()
        findings = []
        checked = 0
        for index, chunk in enumerate(final_chunks):
            if len(chunk.strip()) < 20:
                continue
            result = classifier(chunk[:2000], truncation=True)[0]
            label = str(result.get('label', '')).removeprefix('__label__').casefold()
            score = float(result.get('score', 0))
            checked += 1
            if label != expected and score >= 0.70:
                findings.append({'chunk': index + 1, 'detected': label, 'confidence': round(score, 3)})
        return {
            'test': 'language_id',
            'label': 'Target-language segments',
            'value': len(findings),
            'flagged': bool(findings),
            'details': {
                'expected_language': expected,
                'chunks_checked': checked,
                'wrong_language_segments': findings[:50],
            },
            'note': (
                f'{len(findings)} segment(s) confidently detected as a language other than {LANG_NAMES.get(expected, expected)}: '
                + '; '.join(f"chunk {f['chunk']} → {f['detected']} ({f['confidence']:.0%})" for f in findings[:10])
                if findings else
                f'All {checked} sufficiently long final chunk(s) were classified as {LANG_NAMES.get(expected, expected)} or were low-confidence.'
            ),
        }

    def eval_llm_judge_stage2(
        self, original_chunks: List[str], draft_chunks: List[str], final_chunks: List[str],
        source_lang: str, target_lang: str,
    ) -> Dict:
        """Pairwise draft vs final, on two separate questions, with the
        source in front of the judge.

        This test used to show the judge two target-language passages and ask
        which read better — no source, "purely on naturalness, style and
        tone". That measures exactly the axis a rewriting refinement pass
        optimises, so it reported the pass as harmless while adequacy fell.
        Accuracy is now asked first and separately, and a final that reads
        better but says less shows up as a split verdict instead of a win.

        Chunk-aligned, not paragraph-aligned: Stage 2 stores final_chunks, so
        draft and final can be compared at the same granularity the Stage 1
        judge uses.
        """
        length = min(len(original_chunks), len(draft_chunks), len(final_chunks))
        if length == 0:
            return {
                'test': 'llm_judge_stage2',
                'label': 'LLM judge — draft vs final',
                'value': None,
                'flagged': True,
                'note': (
                    'Nothing to compare. This test needs per-chunk draft and final text; '
                    're-run Continue if this translation was refined by an older version.'
                ),
            }

        source_name = LANG_NAMES.get(source_lang, source_lang)
        target_name = LANG_NAMES.get(target_lang, target_lang)
        indices = self._risk_ranked_indices(original_chunks[:length], self.EVAL_SAMPLE_SIZE)
        tally = {
            'accuracy': Counter(),
            'readability': Counter(),
        }
        samples_used = 0

        for idx in indices:
            original, draft, final = original_chunks[idx], draft_chunks[idx], final_chunks[idx]
            if not draft.strip() or not final.strip() or not original.strip():
                continue
            if draft == final:
                # Nothing was changed here, so there is nothing to judge —
                # counting it as a tie would pad the result with agreement
                # the judge never actually expressed.
                continue

            swap = random.random() < 0.5
            version_a, version_b = (final, draft) if swap else (draft, final)
            prompt = prompts.render(
                'quality/pairwise_editor',
                source_name=source_name, target_name=target_name,
                original=original, version_a=version_a, version_b=version_b,
            )
            raw = self._call_model(prompt)
            samples_used += 1
            if not raw:
                continue

            for axis, pattern in (('accuracy', r'ACCURACY:\s*(A|B|TIE)'), ('readability', r'READABILITY:\s*(A|B|TIE)')):
                match = re.search(pattern, raw.upper())
                if not match:
                    continue
                letter = match.group(1)
                if letter == 'A':
                    tally[axis]['final' if swap else 'draft'] += 1
                elif letter == 'B':
                    tally[axis]['draft' if swap else 'final'] += 1
                else:
                    tally[axis]['tie'] += 1

        if not samples_used:
            return {
                'test': 'llm_judge_stage2',
                'label': 'LLM judge — draft vs final',
                'value': 0,
                'details': {'samples': 0},
                'flagged': False,
                'note': (
                    'Refinement left every sampled chunk byte-identical, so there was '
                    'nothing to compare.'
                ),
            }

        accuracy, readability = tally['accuracy'], tally['readability']
        details = {
            'samples': samples_used,
            'accuracy': {'final_wins': accuracy['final'], 'draft_wins': accuracy['draft'], 'ties': accuracy['tie']},
            'readability': {'final_wins': readability['final'], 'draft_wins': readability['draft'], 'ties': readability['tie']},
        }
        # The failure this is here to catch: the final wins on readability
        # while losing on accuracy. That is a refinement pass buying polish
        # with meaning, and it must not read as a pass.
        traded_meaning_for_polish = (
            accuracy['draft'] > accuracy['final'] and readability['final'] >= readability['draft']
        )
        note = (
            f"Accuracy: final {accuracy['final']}, draft {accuracy['draft']}, tie {accuracy['tie']}. "
            f"Readability: final {readability['final']}, draft {readability['draft']}, tie {readability['tie']}. "
            f"({samples_used} changed chunk(s) sampled.)"
        )
        if traded_meaning_for_polish:
            note += (
                ' The draft is more accurate while the final reads better — refinement '
                'is trading meaning for polish. Ship the draft unless you can see why not.'
            )
        return {
            'test': 'llm_judge_stage2',
            'label': 'LLM judge — draft vs final',
            'value': accuracy['final'],
            'details': details,
            'flagged': traded_meaning_for_polish or accuracy['draft'] > accuracy['final'],
            'note': note,
        }

    # -- Stage 1 tests: is the draft itself an adequate, fluent translation? --

    @staticmethod
    def _adequacy_fluency_prompt(source_name: str, target_name: str, original: str, candidate: str) -> str:
        return prompts.render(
            'quality/adequacy_fluency',
            source_name=source_name, target_name=target_name,
            original=original, candidate=candidate,
        )

    def _score_adequacy_fluency(
        self, pairs: List[Tuple[str, str]], source_name: str, target_name: str,
    ) -> Tuple[List[int], List[int], int]:
        """Runs the adequacy/fluency judge prompt over (original, candidate)
        pairs. Shared by the Stage 1 draft judge and the final-vs-original
        judge — same rubric, different candidate text."""
        adequacy_scores, fluency_scores, samples_used = [], [], 0
        for original, candidate in pairs:
            if not original.strip() or not candidate.strip():
                continue
            prompt = self._adequacy_fluency_prompt(source_name, target_name, original, candidate)
            raw = self._call_model(prompt)
            samples_used += 1
            if not raw:
                continue
            adequacy = re.search(r'ADEQUACY:\s*(\d)', raw)
            fluency = re.search(r'FLUENCY:\s*(\d)', raw)
            if adequacy:
                adequacy_scores.append(int(adequacy.group(1)))
            if fluency:
                fluency_scores.append(int(fluency.group(1)))
        return adequacy_scores, fluency_scores, samples_used

    def eval_llm_judge_stage1(
        self, original_chunks: List[str], draft_chunks: List[str],
        source_lang: str, target_lang: str,
    ) -> Dict:
        return self._judge_adequacy_fluency(
            'llm_judge_stage1', 'LLM judge — adequacy & fluency (draft)',
            original_chunks, draft_chunks, source_lang, target_lang, 'draft',
        )

    def eval_llm_judge_final(
        self, original_chunks: List[str], final_chunks: List[str],
        source_lang: str, target_lang: str,
    ) -> Dict:
        """The same rubric as the Stage 1 judge, scored on the FINAL text.

        Deliberately identical in every respect except which translation is
        being scored — same prompt, same sampling, same chunk boundaries — so
        that the difference between the two numbers means something. Scoring
        the draft by chunk and the final by paragraph position, as this did
        before, produced two numbers on two different texts and an
        "adequacy regression" that was partly an artifact of the alignment.
        """
        return self._judge_adequacy_fluency(
            'llm_judge_final', 'LLM judge — adequacy & fluency (final)',
            original_chunks, final_chunks, source_lang, target_lang, 'final',
        )

    def _judge_adequacy_fluency(
        self, test_name: str, label: str,
        original_chunks: List[str], candidate_chunks: List[str],
        source_lang: str, target_lang: str, candidate_name: str,
    ) -> Dict:
        length = min(len(original_chunks), len(candidate_chunks))
        if length == 0:
            return {
                'test': test_name,
                'label': label,
                'value': None,
                'flagged': True,
                'note': (
                    f'No per-chunk {candidate_name} text to score. Re-run the pass that '
                    'produces it if this translation predates chunk-level storage.'
                ),
            }

        source_name = LANG_NAMES.get(source_lang, source_lang)
        target_name = LANG_NAMES.get(target_lang, target_lang)
        # Deterministic, and the same for the draft judge and the final judge,
        # which is what makes their two scores subtractable.
        indices = self._risk_ranked_indices(original_chunks[:length], self.EVAL_SAMPLE_SIZE)
        pairs = [(original_chunks[idx], candidate_chunks[idx]) for idx in indices]
        adequacy_scores, fluency_scores, samples_used = self._score_adequacy_fluency(pairs, source_name, target_name)

        if not adequacy_scores and not fluency_scores:
            return {
                'test': test_name,
                'label': label,
                'value': None,
                'flagged': True,
                'note': 'The judge model did not return a usable score for any sampled chunk.',
            }

        avg_adequacy = round(mean(adequacy_scores), 2) if adequacy_scores else None
        avg_fluency = round(mean(fluency_scores), 2) if fluency_scores else None
        return {
            'test': test_name,
            'label': label,
            'value': avg_adequacy,
            'details': {
                'avg_adequacy': avg_adequacy,
                'avg_fluency': avg_fluency,
                'samples': samples_used,
                'sampled_chunks': indices,
                'scored': candidate_name,
            },
            'flagged': (avg_adequacy is not None and avg_adequacy < 3) or (avg_fluency is not None and avg_fluency < 3),
            'note': (
                f'{candidate_name.title()}: adequacy {avg_adequacy}/5, fluency {avg_fluency}/5 '
                f'over {samples_used} sampled chunk(s). Both numbers are averages of five '
                f'1–5 ratings, so treat differences under about 0.5 as noise.'
            ),
        }

    def eval_backtranslation_chrf(
        self, original_chunks: List[str], draft_chunks: List[str],
        source_lang: str, target_lang: str,
    ) -> Dict:
        if sacrebleu is None:
            return {
                'test': 'backtranslation_chrf',
                'label': 'Backtranslation chrF',
                'value': None,
                'flagged': True,
                'note': 'sacrebleu is not installed — run: pip install -r requirements.txt',
            }

        indices = self._sample_indices(len(original_chunks), self.EVAL_SAMPLE_SIZE)
        scores = []
        for idx in indices:
            original = original_chunks[idx]
            draft = draft_chunks[idx]
            if not original.strip() or not draft.strip():
                continue
            back_translation, warning = self.stage1_primary_translation(
                draft, source_lang=target_lang, target_lang=source_lang,
            )
            if warning:
                continue
            scores.append(sacrebleu.sentence_chrf(back_translation, [original]).score)

        if not scores:
            return {
                'test': 'backtranslation_chrf',
                'label': 'Backtranslation chrF',
                'value': None,
                'flagged': True,
                'note': 'Backtranslation did not produce a usable result for any sampled chunk.',
            }

        avg = round(mean(scores), 1)
        return {
            'test': 'backtranslation_chrf',
            'label': 'Backtranslation chrF (diagnostic only)',
            'value': avg,
            'details': {'samples': len(scores), 'per_sample': [round(s, 1) for s in scores], 'diagnostic_only': True},
            'flagged': avg < 40,
            'note': (
                f'chrF {avg}/100 over {len(scores)} sampled chunk(s). Diagnostic only, and '
                'not a quality score: chrF measures character overlap with a reference, and '
                'a back-translation is not a reference. The number is the combined error of '
                'the forward and reverse translations, so it cannot be read as the quality '
                'of either. Useful for spotting a chunk that came back as something '
                'completely different; useless for comparing runs.'
            ),
        }

    COMET_KIWI_MODEL = 'Unbabel/wmt22-cometkiwi-da'

    def eval_comet_kiwi(
        self, original_chunks: List[str], candidate_chunks: List[str],
        candidate_name: str = 'draft',
    ) -> Dict:
        """Reference-free neural QE, over whichever translation is handed to
        it — the draft after Start, the final after Continue. Which one was
        scored is reported, because a QE number with no stated subject is how
        a panel ends up showing the draft's score next to a shipped final.

        unbabel-comet is a base requirement, so the import below is a guard
        against a half-built venv rather than the normal path. The multi-GB
        checkpoint is still fetched here, on the first run of this test.

        Downloads the checkpoint via huggingface_hub directly (rather than
        through comet.models.download_model) because that helper swallows
        the original error on a gated/access-denied repo and re-raises a
        generic "not supported by COMET" KeyError with no way to tell that
        apart from a real problem — this repo IS gated on Hugging Face, so
        that failure mode is the common case, not an edge case.
        """
        try:
            import torch
            from comet import load_from_checkpoint
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import GatedRepoError
        except ImportError:
            return {
                'test': 'comet_kiwi',
                'label': 'COMET-Kiwi (reference-free QE)',
                'value': None,
                'flagged': True,
                'note': (
                    'unbabel-comet is missing from the environment — the venv is '
                    'incomplete. Run: ./venv/bin/python -m pip install -r requirements.txt'
                ),
            }

        try:
            length = min(len(original_chunks), len(candidate_chunks))
            indices = self._risk_ranked_indices(original_chunks[:length], self.EVAL_SAMPLE_SIZE)
            data = [
                {'src': original_chunks[idx], 'mt': candidate_chunks[idx]}
                for idx in indices
                if original_chunks[idx].strip() and candidate_chunks[idx].strip()
            ]
            if not data:
                raise ValueError('No chunks available to score')

            try:
                model_dir = snapshot_download(repo_id=self.COMET_KIWI_MODEL)
            except GatedRepoError:
                return {
                    'test': 'comet_kiwi',
                    'label': 'COMET-Kiwi (reference-free QE)',
                    'value': None,
                    'flagged': True,
                    'note': (
                        f'COMET-Kiwi ({self.COMET_KIWI_MODEL}) is a gated Hugging Face model — '
                        f'request access at https://huggingface.co/{self.COMET_KIWI_MODEL}, then run '
                        '`huggingface-cli login` (or set the HF_TOKEN env var) and try again.'
                    ),
                }

            checkpoint_path = os.path.join(model_dir, 'checkpoints', 'model.ckpt')
            model = load_from_checkpoint(checkpoint_path)
            # unbabel-comet's DataLoader setup (comet/models/base.py) always
            # passes multiprocessing_context="fork" when MPS is available,
            # but defaults num_workers to 0 for gpus=0 — a combination
            # PyTorch's DataLoader rejects outright, so predict() throws on
            # every Apple Silicon Mac regardless of sample count. Forcing
            # num_workers>0 to satisfy that isn't a fix either: forking a
            # process that already has a loaded HF fast tokenizer breaks the
            # tokenizer in the child (AttributeError inside the worker). The
            # only combination that actually works — no multiprocessing at
            # all — is what comet already does on every other platform, so
            # make it think MPS isn't available for the scope of this one
            # CPU-only (gpus=0) call.
            mps_is_available = torch.backends.mps.is_available
            torch.backends.mps.is_available = lambda: False
            try:
                output = model.predict(data, batch_size=4, gpus=0)
            finally:
                torch.backends.mps.is_available = mps_is_available
            avg = round(float(output.system_score) * 100, 1)  # type: ignore[attr-defined]
            return {
                'test': 'comet_kiwi',
                'label': 'COMET-Kiwi (reference-free QE)',
                'value': avg,
                'details': {'samples': len(data), 'scored': candidate_name, 'sampled_chunks': indices},
                'flagged': avg < 60,
                'note': (
                    f'COMET-Kiwi {avg}/100 for the {candidate_name} over {len(data)} sampled '
                    f'chunk(s) (0–100, higher is better). Sentence-level by design, so it is '
                    f'blind to anything that spans chunks.'
                ),
            }
        except Exception as e:
            logger.api_logger.error(f"COMET-Kiwi evaluation failed: {e}")
            return {
                'test': 'comet_kiwi',
                'label': 'COMET-Kiwi (reference-free QE)',
                'value': None,
                'flagged': True,
                'note': f'COMET-Kiwi failed: {e}',
            }
