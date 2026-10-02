from pathlib import Path

import pytest

import translator as app_module
from translator import GlossaryTerm, TerminologyManager


def test_parses_arrow_and_tsv_formats():
    manager = TerminologyManager.from_text(
        """
        Mr. Darcy => мистер Дарси | exact
        machine learning\tmaschinelles Lernen\tinflectable
        Home => Heimat
        """
    )

    assert manager.terms == [
        GlossaryTerm("Mr. Darcy", "мистер Дарси", "exact"),
        GlossaryTerm("machine learning", "maschinelles Lernen", "inflectable"),
        GlossaryTerm("Home", "Heimat", "inflectable"),
    ]


def test_a_note_after_the_mode_is_the_authors_own_annotation():
    """The reason the note exists: a rendering alone cannot say which of two
    gendered forms a name takes, and nothing in the source always settles it."""
    manager = TerminologyManager.from_text(
        "Rom => Rom | inflectable {c'est un garçon de huit ans}"
    )

    assert manager.terms == [
        GlossaryTerm("Rom", "Rom", "inflectable", "c'est un garçon de huit ans"),
    ]


def test_a_note_may_hold_the_delimiters_the_entry_itself_uses():
    """A note is prose. It must not have to avoid `=>` or `|` to survive."""
    manager = TerminologyManager.from_text(
        "a | b => c => d | preferred {x | y => z}"
    )

    assert manager.terms == [
        GlossaryTerm("a | b", "c => d", "preferred", "x | y => z"),
    ]


def test_a_note_also_follows_a_tab_separated_row():
    manager = TerminologyManager.from_text(
        "machine learning\tmaschinelles Lernen\tinflectable\t{masculine}"
    )

    assert manager.terms == [
        GlossaryTerm(
            "machine learning", "maschinelles Lernen", "inflectable", "masculine",
        ),
    ]


def test_an_entry_without_a_note_still_parses_exactly_as_before():
    assert TerminologyManager.from_text(
        "Rom => Rom | inflectable\nRomel => Romel\n"
    ).terms == [
        GlossaryTerm("Rom", "Rom", "inflectable"),
        GlossaryTerm("Romel", "Romel", "inflectable"),
    ]


def test_a_malformed_note_is_rejected_rather_than_read_as_a_mode():
    """Without this the note is swallowed into the mode and the entry is
    silently rejected, or worse, accepted as a mode nobody can enforce."""
    for text, message in (
        ("Rom => Rom | inflectable {}", "the note is empty"),
        ("Rom => Rom | inflectable {a boy", "unbalanced brace"),
        ("Rom => Rom | inflectable {a boy} and more", "unbalanced brace"),
        ("Rom => Rom | inflectable {" + "x" * 201 + "}", "note exceeds"),
    ):
        with pytest.raises(ValueError, match=message):
            TerminologyManager.from_text(text)


def test_a_note_survives_the_round_trip_through_the_text_format():
    """Every place that refills the textarea from stored terms goes through
    this, so a note written once is not lost on the next reopen."""
    line = TerminologyManager.format_line(
        "Rom", "Rom", "inflectable", "c'est un garçon",
    )

    assert line == "Rom => Rom | inflectable {c'est un garçon}"
    assert TerminologyManager.from_text(line).terms == [
        GlossaryTerm("Rom", "Rom", "inflectable", "c'est un garçon"),
    ]


def test_the_browser_parses_notes_by_the_same_rules():
    """The editor validates in JavaScript, before anything is uploaded, and
    there is no JS test runner here. If the two drift, the status line calls a
    perfectly good entry malformed — or worse, calls a broken one valid — and
    the author has no way to tell which parser Start will use."""
    index_html = (
        Path(app_module.__file__).parent / 'static' / 'index.html'
    ).read_text(encoding='utf-8')

    assert "const GLOSSARY_NOTE_SUFFIX = /\\{([^{}]*)\\}\\s*$/;" in index_html
    assert 'const GLOSSARY_MAX_NOTE_LENGTH = 200;' in index_html
    assert "'the note is empty'" in index_html
    assert "'an unbalanced brace" in index_html
    assert 'a note exceeds ${GLOSSARY_MAX_NOTE_LENGTH} characters' in index_html


def test_an_entry_with_no_note_is_formatted_exactly_as_it_was():
    assert TerminologyManager.format_line("Rom", "Rom", "inflectable") == (
        "Rom => Rom | inflectable"
    )


def test_omitted_mode_defaults_to_inflectable_for_arrow_and_tsv_formats():
    manager = TerminologyManager.from_text(
        "home => дом\nmachine learning\tмашинное обучение"
    )

    assert manager.terms == [
        GlossaryTerm("home", "дом", "inflectable"),
        GlossaryTerm("machine learning", "машинное обучение", "inflectable"),
    ]


def test_relevance_is_case_insensitive_and_supports_cjk():
    manager = TerminologyManager.from_text(
        """
        ALICE => Алиса | exact
        人工知能 => artificial intelligence | exact
        """
    )

    assert [term.source for term in manager.relevant_terms("Alice met Bob.")] == [
        "ALICE"
    ]
    assert [term.source for term in manager.relevant_terms("人工知能の研究")] == [
        "人工知能"
    ]


def test_exact_violations_only_check_relevant_exact_terms():
    manager = TerminologyManager.from_text(
        """
        garden => сад | exact
        house => дом | preferred
        absent => отсутствует | exact
        """
    )

    assert manager.exact_violations("The garden and house.", "Сад и жилище.") == []
    assert manager.exact_violations("The garden and house.", "Двор и жилище.") == [
        {"source": "garden", "required_target": "сад"}
    ]


def test_exact_terms_replace_only_literal_source_leaks():
    manager = TerminologyManager.from_text(
        "Dursley => Дурсль | exact\nHome => дом | inflectable"
    )

    translated, replacements = manager.enforce_exact_source_forms(
        "Dursley arrived. The Dursleys stayed. Home remained untranslated."
    )

    assert translated == "Дурсль arrived. The Dursleys stayed. Home remained untranslated."
    assert replacements == [{"source": "Dursley", "target": "Дурсль", "count": 1}]


def test_exact_source_replacement_is_case_insensitive_but_respects_word_boundaries():
    manager = TerminologyManager.from_text("garden => сад | exact")

    translated, replacements = manager.enforce_exact_source_forms(
        "GARDEN, garden; gardener; gardens."
    )

    assert translated == "сад, сад; gardener; gardens."
    assert replacements == [{"source": "garden", "target": "сад", "count": 2}]


def test_fingerprint_is_stable_but_changes_with_constraints():
    first = TerminologyManager.from_text(
        "cat => кот | exact\ndog => пёс | preferred"
    )
    reordered = TerminologyManager.from_text(
        "dog => пёс | preferred\ncat => кот | exact"
    )
    changed = TerminologyManager.from_text(
        "cat => кошка | exact\ndog => пёс | preferred"
    )

    assert first.fingerprint() == reordered.fingerprint()
    assert first.fingerprint() != changed.fingerprint()


def test_editing_only_a_note_retires_the_cached_chunks():
    """The note is part of the prompt, so it has to be part of the cache key.
    Without it, correcting a wrong note and pressing Continue would hand back
    the chunks translated under the note that was just fixed."""
    before = TerminologyManager.from_text("Rom => Rom | inflectable {a girl}")
    after = TerminologyManager.from_text("Rom => Rom | inflectable {a boy}")

    assert before.fingerprint() != after.fingerprint()
