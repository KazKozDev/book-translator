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
