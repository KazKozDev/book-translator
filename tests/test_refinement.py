"""Stage 2: the estimate/patch half that runs without a model.

The model's answer is untrusted input here — it arrives as prose-wrapped
JSON from a quantised local model and may name spans that do not exist. What
these tests pin down is that nothing unverifiable ever reaches the text.
"""

import json

import pytest

from translator import BookTranslator


DRAFT = 'Мистер Дурсли был директором фирмы Grunnings, которая делала свёрла.'


def test_parses_json_wrapped_in_prose_and_fences():
    raw = 'Sure! Here are the errors I found:\n```json\n[{"span": "x"}]\n```\nHope that helps.'

    assert BookTranslator._parse_json_array(raw) == [{'span': 'x'}]


def test_unparseable_answers_yield_no_errors():
    for raw in (None, '', 'I could not find any problems.', '[not json]', '{not json}'):
        assert BookTranslator._parse_json_array(raw) == []


def test_a_lone_object_is_read_rather_than_discarded():
    """One error, written without the array around it, is still one error.
    Nothing skips validation by arriving this way: every span is still
    checked against the draft before it can be patched in."""
    assert BookTranslator._parse_json_array('{"span": "x"}') == [{'span': 'x'}]


def test_only_spans_present_in_the_draft_survive_validation():
    errors = BookTranslator.validate_estimate_spans(
        [
            {'span': 'Grunnings', 'type': 'terminology', 'severity': 'major', 'replacement': 'Граннингс'},
            # Re-typed from memory rather than copied — no position to patch.
            {'span': 'Мистер Дурсль', 'type': 'consistency', 'severity': 'major', 'replacement': 'Мистер Дурсль'},
            # No replacement to apply.
            {'span': 'свёрла', 'type': 'mistranslation', 'severity': 'minor', 'replacement': ''},
        ],
        DRAFT,
    )

    assert [error['span'] for error in errors] == ['Grunnings']


def test_severity_orders_errors_and_unknown_categories_are_normalised():
    errors = BookTranslator.validate_estimate_spans(
        [
            {'span': 'свёрла', 'type': 'nonsense-category', 'severity': 'minor', 'replacement': 'дрели'},
            {'span': 'Grunnings', 'type': 'accuracy', 'severity': 'critical', 'replacement': 'Граннингс'},
        ],
        DRAFT,
    )

    assert [error['span'] for error in errors] == ['Grunnings', 'свёрла']
    assert errors[0]['type'] == 'mistranslation'  # aliased
    assert errors[1]['type'] == 'other'  # unrecognised, kept but not objective


def test_a_no_op_replacement_is_dropped():
    assert BookTranslator.validate_estimate_spans(
        [{'span': 'свёрла', 'type': 'style', 'severity': 'minor', 'replacement': 'свёрла'}],
        DRAFT,
    ) == []


def test_patch_touches_only_the_reported_spans():
    patched, applied = BookTranslator.stage2_patch(
        DRAFT,
        [
            {'span': 'Grunnings', 'replacement': 'Граннингс', 'type': 'terminology', 'severity': 'major'},
            {'span': 'свёрла', 'replacement': 'дрели', 'type': 'mistranslation', 'severity': 'major'},
        ],
    )

    assert patched == 'Мистер Дурсли был директором фирмы Граннингс, которая делала дрели.'
    assert len(applied) == 2
    # Everything outside the two spans is carried over character for
    # character — this is what keeps the draft/final diff small.
    assert patched.startswith('Мистер Дурсли был директором фирмы ')


def test_overlapping_spans_do_not_corrupt_the_text():
    draft = 'один два три'
    patched, applied = BookTranslator.stage2_patch(
        draft,
        [
            {'span': 'один два', 'replacement': 'ONE TWO', 'type': 'style', 'severity': 'major'},
            {'span': 'два три', 'replacement': 'TWO THREE', 'type': 'style', 'severity': 'major'},
        ],
    )

    assert patched == 'ONE TWO три'
    assert len(applied) == 1


def test_a_repeated_span_is_patched_once_per_reported_error():
    patched, applied = BookTranslator.stage2_patch(
        'кот и кот',
        [{'span': 'кот', 'replacement': 'пёс', 'type': 'style', 'severity': 'minor'}],
    )

    assert patched == 'пёс и кот'
    assert len(applied) == 1


def test_no_errors_leaves_the_draft_identical():
    patched, applied = BookTranslator.stage2_patch(DRAFT, [])

    assert patched == DRAFT
    assert applied == []


def test_style_edits_never_reach_the_text():
    """The observed failure: the review pass swapped a perfectly good verb
    for a longer synonym and the verifier waved it through, because on the
    style axis there is nothing to be wrong about."""
    assert not BookTranslator.is_actionable_error(
        {'type': 'style', 'severity': 'major', 'span': 'x', 'replacement': 'y'}
    )
    assert not BookTranslator.is_actionable_error(
        {'type': 'other', 'severity': 'critical', 'span': 'x', 'replacement': 'y'}
    )


def test_minor_subjective_errors_are_reported_but_not_applied():
    assert not BookTranslator.is_actionable_error(
        {'type': 'mistranslation', 'severity': 'minor', 'span': 'x', 'replacement': 'y'}
    )
    assert BookTranslator.is_actionable_error(
        {'type': 'mistranslation', 'severity': 'major', 'span': 'x', 'replacement': 'y'}
    )


def test_glossary_fixes_are_applied_at_any_severity():
    """A required rendering being absent is a fact, not a matter of degree."""
    for severity in ('critical', 'major', 'minor'):
        assert BookTranslator.is_actionable_error(
            {'type': 'terminology', 'severity': severity, 'span': 'x', 'replacement': 'y'}
        )
        assert BookTranslator.is_actionable_error(
            {'type': 'consistency', 'severity': severity, 'span': 'x', 'replacement': 'y'}
        )


@pytest.mark.parametrize('span', ['', 'x', '   '])
def test_spans_too_short_to_locate_are_rejected(span):
    assert BookTranslator.validate_estimate_spans(
        [{'span': span, 'type': 'style', 'severity': 'minor', 'replacement': 'что-то'}],
        DRAFT,
    ) == []


# -- Replacements that are the source text, or the absence of text.
#
# The observed failure, on the opening page of a chapter: a display block
# ("WANTED / HER / TO / BE / MAD / AT / ME") had been rendered as French
# prose by Stage 1, and the review pass reported the page as a run of
# `omission` errors whose replacements were the English source sentences,
# copied verbatim. `omission` is judge-exempt, so the patch was applied
# unverified and a third of the chapter went back to English. The draft was
# correct the whole time. These cases pin the guard that keeps it.

FR_PAGE_SOURCE = (
    '27. Rome\n\nROME\n\nNew York\n\nWANTED\n\nHER\n\nTO\n\nBE\n\nMAD\n\nAT\n\nME\n\n.\n\n'
    'It would have been better if she was mad at me.\n\n'
    'Anything would’ve been preferable to the way she iced over, blankness '
    'behind her eyes as she regarded me. “I can’t this weekend, Rome. I have '
    'so much I need to get done before Monday.”\n\nGreat.'
)
FR_PAGE_DRAFT = (
    '27. Rome\n\nROME\n\nELLE\n\nDOIT\n\nÊTRE\n\nEN\n\nCOLÈRE\n\nCONTRE\n\nMOI.\n\n'
    'Il aurait été préférable qu’elle soit en colère contre moi.\n\n'
    'N’importe quoi aurait été mieux que cette froideur, ce vide dans son '
    'regard lorsqu’elle me regardait. « Je ne peux pas ce week-end, Rome. '
    'J’ai tellement de choses à faire avant lundi. »\n\nSuper.'
)


def test_a_replacement_that_is_the_source_sentence_is_dropped():
    """The failure itself: the review pass hands back the English in place of
    French that was already there, and the draft is left alone."""
    errors = BookTranslator.validate_estimate_spans(
        [{
            'span': 'Il aurait été préférable qu’elle soit en colère contre moi.',
            'type': 'omission', 'severity': 'major',
            'replacement': 'It would have been better if she was mad at me.',
        }],
        FR_PAGE_DRAFT, FR_PAGE_SOURCE,
    )

    assert errors == []


def test_source_copies_are_dropped_however_they_are_punctuated_or_cased():
    """Quotation style and case are what a model changes while copying. They
    must not be what lets a copy through."""
    for replacement in (
        'anything would’ve been preferable to the way she iced over, blankness'
        ' behind her eyes as she regarded me.',
        'ANYTHING WOULD’VE BEEN PREFERABLE TO THE WAY SHE ICED OVER, BLANKNESS'
        ' BEHIND HER EYES AS SHE REGARDED ME.',
        'Anything would’ve been preferable to the way she iced over, blankness'
        ' behind her eyes as she regarded me.',
    ):
        assert BookTranslator.validate_estimate_spans(
            [{
                'span': 'N’importe quoi aurait été mieux que cette froideur, ce vide'
                ' dans son regard lorsqu’elle me regardait.',
                'type': 'omission', 'severity': 'major', 'replacement': replacement,
            }],
            FR_PAGE_DRAFT, FR_PAGE_SOURCE,
        ) == []


def test_a_name_that_belongs_in_the_target_text_is_not_treated_as_a_copy():
    """The guard is length-deliberate. "Rome", "New York" and "Ivy" are all in
    the source and all correct in a French page — a proper noun is never long
    enough to be mistaken for a copied sentence."""
    errors = BookTranslator.validate_estimate_spans(
        [{'span': 'CONTRE', 'type': 'terminology', 'severity': 'critical',
          'replacement': 'New York'}],
        FR_PAGE_DRAFT, FR_PAGE_SOURCE,
    )

    assert [error['replacement'] for error in errors] == ['New York']


def test_a_source_sentence_appended_to_the_span_is_still_a_copy():
    """The second shape the review pass produces, taken from the other book
    that hit it: the words that were already there, kept, with an English
    source sentence written on after them. The whole replacement is not a
    substring of the source, so a whole-string comparison misses it — the run
    check is what catches it."""
    source = (
        'Poor Easton with that horrible mother of his. I think the whole town '
        'celebrated our arrival that day.'
    )
    draft = 'Pauvre Easton avec cette horrible mère. Lars est enfin remis.'
    errors = BookTranslator.validate_estimate_spans(
        [{
            'span': 'Pauvre Easton avec cette horrible mère.',
            'type': 'omission', 'severity': 'major',
            'replacement': 'Pauvre Easton avec cette horrible mère.  Poor Easton'
                           ' with that horrible mother of his. I think the whole'
                           ' town celebrated our arrival that day.',
        }],
        draft, source,
    )

    assert errors == []


def test_a_glossary_fix_may_not_put_a_source_word_back():
    """A `terminology` or `consistency` fix whose replacement is a single word
    of the source is not a rendering — it is the original word. The
    chapter-opening page in the observed failure produced three of these in a
    row ("ELLE" => "New York", "DOIT" => "WANTED"), each one word and so
    invisible to the run check."""
    for replacement in ('WANTED', 'ME', 'BE'):
        assert BookTranslator.validate_estimate_spans(
            [{'span': 'ELLE', 'type': 'terminology', 'severity': 'critical',
              'replacement': replacement}],
            FR_PAGE_DRAFT, FR_PAGE_SOURCE,
        ) == []


def test_a_multiword_name_the_source_also_contains_is_still_allowed():
    """The check is deliberately strict — a single word only. "New York" is
    two, and a place name is normally left as it is in a French page."""
    errors = BookTranslator.validate_estimate_spans(
        [{'span': 'CONTRE', 'type': 'terminology', 'severity': 'critical',
          'replacement': 'New York'}],
        FR_PAGE_DRAFT, FR_PAGE_SOURCE,
    )

    assert len(errors) == 1


def test_a_kept_name_is_not_a_source_word_when_the_glossary_agreed_to_keep_it():
    """`Rom => Rom | inflectable {c'est un garçon}` is the case a note exists
    for, and an English name kept in a French page is the same thing. Without
    the agreed-rendering exemption the guard refused every single-word kept name
    in the book — `Rom`, `Ivy`, `Mr`, `Darcy` all came back True against a
    source containing them — which is the failure it was written to prevent
    arriving from the other direction."""
    source = 'Rom was in the garden. Ivy was there too. Mr. Darcy arrived.'
    draft = 'Rom était dans le jardin. Ivy était là aussi.'

    for replacement in ('Rom', 'Ivy'):
        assert len(BookTranslator.validate_estimate_spans(
            [{'span': 'Rom était', 'type': 'terminology', 'severity': 'critical',
              'replacement': replacement}],
            draft, source, agreed_targets={'Rom', 'Ivy', 'Mr. Darcy'},
        )) == 1, replacement


def test_a_bare_title_is_still_refused_when_the_agreed_rendering_is_the_full_name():
    """The exemption compares the whole replacement to the agreed rendering, so
    it does not partially match: `Mr. Darcy` is two words and a replacement of
    `Mr` alone is still the source word put back, not the agreed name."""
    assert BookTranslator.validate_estimate_spans(
        [{'span': 'Elle a dit', 'type': 'terminology', 'severity': 'critical',
          'replacement': 'Mr'}],
        'Elle a dit que le train était en retard.',
        'She said Mr. Darcy had arrived.',
        agreed_targets={'Mr. Darcy'},
    ) == []


def test_the_exemption_does_not_spread_to_names_the_glossary_says_nothing_about():
    """Only the terms relevant to this chunk are passed in, so the guard defers
    to exactly the contract the reviewer was shown. A source word that no term
    claims is still refused — which is the "ELLE" => "WANTED" case."""
    source = 'Rom was in the garden. Ivy was there too. Mr. Darcy arrived.'
    draft = 'Rom était dans le jardin. Ivy était là aussi.'

    for replacement in ('Ivy', 'Mr', 'Darcy'):
        assert BookTranslator.validate_estimate_spans(
            [{'span': 'Rom était', 'type': 'terminology', 'severity': 'critical',
              'replacement': replacement}],
            draft, source, agreed_targets={'Rom'},
        ) == [], replacement


def test_a_source_word_that_is_also_an_agreed_rendering_is_a_rendering():
    """The exemption is not a loophole: if the contract says this name is
    written `WANTED`, then writing `WANTED` is the fix, not the source word put
    back. That holds for every mode, `preferred` included — a preferred wording
    is still the agreed one, and refusing it would be refusing the glossary."""
    errors = BookTranslator.validate_estimate_spans(
        [{'span': 'ELLE', 'type': 'terminology', 'severity': 'critical',
          'replacement': 'WANTED'}],
        FR_PAGE_DRAFT, FR_PAGE_SOURCE, agreed_targets={'WANTED', 'ME'},
    )

    assert len(errors) == 1


def test_the_exemption_ignores_case_and_accents_like_every_other_comparison():
    source = 'ROM was in the garden.'
    draft = 'ROM était dans le jardin.'
    errors = BookTranslator.validate_estimate_spans(
        [{'span': 'ROM était', 'type': 'consistency', 'severity': 'critical',
          'replacement': 'rom'}],
        draft, source, agreed_targets={'Rom'},
    )

    assert len(errors) == 1


def test_a_call_that_knows_nothing_of_the_glossary_keeps_the_old_behaviour():
    """`agreed_targets` is optional, so a caller holding only a draft is not
    broken — the guard simply has nothing to defer to."""
    assert BookTranslator.validate_estimate_spans(
        [{'span': 'ELLE', 'type': 'terminology', 'severity': 'critical',
          'replacement': 'WANTED'}],
        FR_PAGE_DRAFT, FR_PAGE_SOURCE,
    ) == []


def test_the_agreed_renderings_reach_the_guard_through_the_whole_pass(monkeypatch):
    """The exemption has to survive the two layers between the route and the
    guard, or the fix is invisible in a real run. The same reported error is
    refused without the glossary and kept with it."""
    translator = BookTranslator(model_name='reviewer:12b', verifier_model='verifier:27b')
    reported = json.dumps([
        {'span': 'ELLE', 'type': 'terminology', 'severity': 'critical',
         'replacement': 'WANTED'},
    ])

    def run(agreed):
        _script_model_calls(monkeypatch, [reported])
        _, _, details = translator.stage2_reflection_improvement(
            original_text=FR_PAGE_SOURCE, draft_translation=FR_PAGE_DRAFT,
            source_lang='english', target_lang='french',
            agreed_targets=agreed,
        )
        return details

    refused = run(None)
    assert refused['errors_found'] == 0
    assert refused['dropped_by_guard'] == {'source word': 1}

    kept = run({'WANTED'})
    assert kept['errors_found'] == 1
    assert kept['dropped_by_guard'] == {}


def test_the_source_word_check_does_not_apply_to_other_categories():
    """Only the two categories whose replacement is meant to be a rendering.
    A mistranslation is free to propose any wording, and reporting a source
    word there is not evidence of anything."""
    errors = BookTranslator.validate_estimate_spans(
        [{'span': 'Elle', 'type': 'mistranslation', 'severity': 'major',
          'replacement': 'BE'}],
        'Elle est là. Elle reste.', 'She said BE to her.',
    )

    assert len(errors) == 1


def test_a_replacement_that_restates_the_text_after_its_span_is_dropped():
    """The second shape the review pass produces, and the one a reader sees as
    a repeated paragraph.

    Taken from a real chapter: the span is one sentence, and the replacement
    quotes that sentence and then keeps going, restating the two hundred
    characters that follow it. Patching that in prints those two hundred
    characters twice — a five-sentence paragraph, then the same five
    sentences."""
    draft = (
        '« On m’avait dit que le cottage serait meublé, mais quand je suis '
        'arrivée, il était vide », ai-je exclaimed. « Les gars étaient '
        'contrariés quand ils l’ont découvert. Apparemment, ils avaient mis '
        'les affaires des grands-parents de Finn au grenier avant que je '
        'déménage. Je leur ai dit de ne pas m’acheter de nouvelles choses. »'
    )
    span = '« On m’avait dit que le cottage serait meublé, mais quand je suis arrivée, il était vide », ai-je exclamé.'
    errors = BookTranslator.validate_estimate_spans(
        [{'span': span, 'type': 'omission', 'severity': 'major',
          'replacement': span + ' « Les gars contrariés quand ils l’ont '
                               'découvert. Apparemment, ils avaient mis les '
                               'affaires des grands-parents de Finn au '
                               'grenier avant que je déménage. Je leur ai dit '
                               'de ne pas m’acheter de nouvelles choses. »'}],
        draft,
    )

    assert errors == []


def test_a_whole_sentence_written_in_the_source_language_is_dropped():
    errors = BookTranslator.validate_estimate_spans(
        [{'span': 'BIPIPEZ SI VOUS AIMEZ LES GÂTEAUX', 'type': 'mistranslation',
          'severity': 'major', 'replacement': 'HONK IF YOU LOVE CAKE'}],
        'BIPIPEZ SI VOUS AIMEZ LES GÂTEAUX', '', 'en',
    )

    assert errors == []


def test_quoting_the_span_itself_is_not_a_duplicate():
    """The estimate prompt asks for exactly this when restoring missing
    content: the span, then the content that belongs with it. The span's own
    words are not counted against it — only what follows it in the draft."""
    draft = 'Elle grince depuis des lustres. Je suis sûre qu’il s’en occupe.'
    span = 'Elle grince depuis des lustres.'
    errors = BookTranslator.validate_estimate_spans(
        [{'span': span, 'type': 'omission', 'severity': 'major',
          'replacement': span + ' C’est un bruit constant, à n’en plus douter.'}],
        draft,
    )

    assert len(errors) == 1


def test_an_ordinary_short_fix_is_untouched():
    """The threshold is nine words because the two texts here are the same
    language, where a coincidence of common words is likelier. Below it, a
    real fix is never blocked — measured over the three books in this
    workspace, the guard catches 97 edits that really do duplicate the text
    and blocks 1 that does not."""
    draft = 'Elle est restée là. Il a hoché la tête.'
    for span, replacement in (
        ('Elle est restée là.', 'Elle est restée.'),
        ('Il a hoché la tête.', 'Il a hoché la tête.'),
        ('Elle', 'Celle-là'),
    ):
        if span == replacement:
            continue
        errors = BookTranslator.validate_estimate_spans(
            [{'span': span, 'type': 'mistranslation', 'severity': 'major',
              'replacement': replacement}],
            draft,
        )
        assert len(errors) == 1, (span, replacement)


def test_dropping_a_duplicate_leaves_the_draft_whole():
    """The direction matters. Declining the edit keeps the reader's text
    exactly as it was, so this guard cannot open a hole — the failure it
    prevents and the failure it could have caused are not the same one."""
    draft = (
        'Elle grince depuis des lustres. Je suis sûre qu’il s’en occupe et '
        'qu’elle n’a pas l’intention de réparer quoi que ce soit.'
    )
    span = 'Elle grince depuis des lustres.'
    following = draft[len(span):]
    kept = BookTranslator.validate_estimate_spans(
        [{'span': span, 'type': 'omission', 'severity': 'major',
          'replacement': span + ' ' + following}],
        draft,
    )
    patched, applied = BookTranslator.stage2_patch(draft, kept)

    assert patched == draft
    assert applied == []


def test_a_glossary_rendering_need_not_be_listed_by_this_chunk_s_glossary():
    """A guard that rejected a `terminology` fix whose replacement was absent
    from the glossary was written, measured and removed. Over the three books
    here it blocked 414 edits, two thirds of them `consistency` errors about
    words the glossary has no entry for — real fixes dropped because their
    replacement was missing from an unrelated term's list. Deciding whether a
    span is about a covered term needs the span aligned to the source, which
    this pass does not have, so the judgement stays out of it and the review
    desk keeps it instead."""
    errors = BookTranslator.validate_estimate_spans(
        [{'span': 'garçon', 'type': 'consistency', 'severity': 'major',
          'replacement': 'garçons'},
         {'span': 'petite omègue', 'type': 'terminology', 'severity': 'major',
          'replacement': 'petite omega'}],
        'Un garçon. Une petite omègue.', '', 'en',
    )

    assert [error['replacement'] for error in errors] == ['garçons', 'petite omega']


def test_a_replacement_that_deletes_the_span_is_dropped():
    """The other way to lose the reader's text: a real fix, a real
    translation, but three words standing where a paragraph was."""
    long_span = (
        'N’importe quoi aurait été mieux que cette froideur, ce vide dans son '
        'regard lorsqu’elle me regardait.'
    )
    assert BookTranslator.validate_estimate_spans(
        [{'span': long_span, 'type': 'omission', 'severity': 'major',
          'replacement': 'Super.'}],
        FR_PAGE_DRAFT, FR_PAGE_SOURCE,
    ) == []


def test_a_genuinely_shorter_rendering_is_still_allowed():
    """French runs longer than English, not shorter, so a fix that is somewhat
    more compact is a fix and not a deletion. The guard is a collapse, not a
    comparison."""
    long_span = (
        'N’importe quoi aurait été mieux que cette froideur, ce vide dans son '
        'regard lorsqu’elle me regardait.'
    )
    errors = BookTranslator.validate_estimate_spans(
        [{'span': long_span, 'type': 'omission', 'severity': 'major',
          'replacement': 'Tout aurait été préférable à ce vide dans son regard.'}],
        FR_PAGE_DRAFT, FR_PAGE_SOURCE,
    )

    assert len(errors) == 1


def test_a_whole_refined_page_keeps_the_draft_even_when_nothing_else_is_wrong(monkeypatch):
    """End to end on the real chunk: every source-copying error reported is
    dropped, the French draft is returned untouched, and the verifier is never
    asked — there is nothing left to vote on."""
    translator = BookTranslator(model_name='reviewer:12b', verifier_model='verifier:27b')
    reported = json.dumps([
        {'span': 'Il aurait été préférable qu’elle soit en colère contre moi.',
         'replacement': 'It would have been better if she was mad at me.',
         'type': 'omission', 'severity': 'major'},
        {'span': 'N’importe quoi aurait été mieux que cette froideur, ce vide dans son'
         ' regard lorsqu’elle me regardait.',
         'replacement': 'Anything would’ve been preferable to the way she iced over,'
         ' blankness behind her eyes as she regarded me.',
         'type': 'omission', 'severity': 'major'},
    ])
    calls = _script_model_calls(monkeypatch, [reported])

    text, warning, details = translator.stage2_reflection_improvement(
        original_text=FR_PAGE_SOURCE, draft_translation=FR_PAGE_DRAFT,
        source_lang='english', target_lang='french',
    )

    assert text == FR_PAGE_DRAFT
    assert warning is None
    assert details['errors_found'] == 0
    assert details['errors_applied'] == 0
    assert len(calls) == 1  # estimate only — there was nothing left to verify


def test_omitting_the_source_keeps_the_old_behaviour():
    """`original_text` is optional so a caller holding only a draft is not
    broken by the guard — the copy check simply does not run."""
    errors = BookTranslator.validate_estimate_spans(
        [{'span': 'Super.', 'type': 'omission', 'severity': 'major',
          'replacement': 'Great.'}],
        FR_PAGE_DRAFT,
    )

    assert len(errors) == 1


# -- Replacements written in the wrong language.
#
# The other three guards all compare the replacement against the source text,
# which catches a copy and nothing else. These cases are the review pass
# paraphrasing instead of copying: "« D'où venez-vous ? »" answered with
# "Where are you from?". No run of words is shared with the source, so every
# text comparison passes it, and the English lands in a French page.

def test_an_english_paraphrase_is_dropped_though_it_shares_no_words_with_the_source():
    source = "D'où venez-vous ? Votre porte grince. Comment ça s'est passé ?"
    draft = "« D’où venez-vous ? » « Votre porte grince. » « Comment ça s’est passé ? »"
    errors = BookTranslator.validate_estimate_spans(
        [{'span': '« D’où venez-vous ? »', 'type': 'omission', 'severity': 'major',
          'replacement': 'Where are you from?'}],
        draft, source, 'en',
    )

    assert errors == []


def test_a_sentence_long_enough_to_judge_is_dropped():
    errors = BookTranslator.validate_estimate_spans(
        [{'span': 'J’ai fait taire la classe', 'type': 'omission', 'severity': 'major',
          'replacement': 'Your door is squeaky. Have you asked Bruce to fix it?'}],
        '« Votre porte grince. »', 'Votre porte grince.', 'en',
    )

    assert errors == []


def test_correct_french_is_not_mistaken_for_english():
    """The guard has to be quiet on the ordinary case, or it would drop real
    fixes all day. Measured over 12,506 French paragraphs from the three books
    in this workspace, it fires on 4 — all of them short lines that lean on
    words shared with English ("On a", "on n'a pas de problème")."""
    for replacement in (
        'J’ai fait taire la classe en claquant des mains trois fois',
        'Votre porte grince. Avez-vous demandé à Bruce de la réparer ?',
        'Je pensais que l’ex d’Ivy ne revenait à Starlight Grove que pendant Noël',
    ):
        assert not BookTranslator._is_in_source_language(replacement, 'en')


def test_the_threshold_is_set_for_the_direction_that_is_asked_about():
    """The guard is asked one question — "is this replacement in the source
    language?" — and that is the direction it is tuned for: source text in
    the output, 0.80 against a 0.38 bar. It is not a general language
    identifier, and this pins that down rather than leaving it assumed.

    French prose scores lower against its own function words (0.25-0.36) than
    English does, because French leans on articles and pronouns English drops.
    Asking the reverse question would therefore need a lower bar, and nothing
    in the pipeline asks it: the target language is never the suspect."""

    assert BookTranslator._is_in_source_language('Where are you from now?', 'en')
    assert not BookTranslator._is_in_source_language(
        'J’ai fait taire la classe en claquant des mains trois fois', 'en',
    )


def test_the_bar_sits_above_the_worst_false_positive_that_was_measured():
    """The only corpus measured is one book in one language pair, and it put
    correct French replacements as high as 0.36. A bar under that number is a
    bar that has already been seen to fire on a good translation — and the
    display case that motivated the guard scores 0.40, so there is room."""
    assert BookTranslator.SOURCE_LANGUAGE_MIN_SHARE > 0.36
    assert BookTranslator._is_in_source_language('HONK IF YOU LOVE CAKE', 'en')


def test_a_word_shared_with_the_target_language_is_not_a_marker():
    """The guard asks one direction — source language or not — so a marker
    that is also a correct word of the language being translated into fires
    on good translations and nothing else. Spanish and Portuguese share most of
    their function words; Italian shares a dozen with Spanish."""
    portuguese = 'Ele não sabia se eles queriam falar com ela, mas eles também não.'
    italian = 'Non è mai venuto, e non so perché lui non lo faccia con noi.'

    assert not BookTranslator._is_in_source_language(portuguese, 'es')
    assert not BookTranslator._is_in_source_language(italian, 'es')
    # Each language's own text is still recognised in it.
    assert BookTranslator._is_in_source_language(portuguese, 'pt')
    assert BookTranslator._is_in_source_language(italian, 'it')


def test_a_name_or_a_greeting_is_too_short_to_judge():
    """Four words is the floor. Below it the signal is not reliable, and a
    name kept in English is correct in a French page."""
    for replacement in ('Ivy', 'New York', 'Great.', 'Super.', 'Eh bien'):
        assert not BookTranslator._is_in_source_language(replacement, 'en')


# One ordinary sentence per covered language. Two jobs: each must be recognised
# as itself (recall), and must not be recognised as any other language (the
# guard's whole reason for being one-directional).
SAMPLES = {
    'en': 'He was alone in the big house and did not know what to do.',
    'fr': 'Il etait seul dans la grande maison et ne savait que faire.',
    'de': 'Er war allein in dem grossen Haus und wusste nicht, was er tun sollte.',
    'es': 'Estaba solo en la gran casa y no sabia que hacer.',
    'it': 'Era solo nella grande casa e non sapeva che cosa fare.',
    'pt': 'Ele estava sozinho na grande casa e nao sabia o que fazer.',
    'ru': 'Он был один в большом доме и не знал, что делать.',
    'ko': '그는 큰 집에 혼자 있었고 무엇을 해야 할지 몰랐다.',
}

#: Korean is in the cross-contamination matrix below but not in the recall test.
#: It is space-separated, so the method fits it, but it agglutinates: the
#: particles and verb endings that carry the function words are glued onto the
#: stem, so a token like `있었고` or `해야` matches no list of standalone words.
#: Its 28 markers were not widened for the same reason the bar was not moved:
#: it needs its own measured pass, and guessing at it would be the exact
#: mistake the second pass was made to avoid. Until then a Korean source is
#: recognised by the run check rather than by the word list.
RECALL_LANGUAGES = ('en', 'fr', 'de', 'es', 'it', 'pt', 'ru')


@pytest.mark.parametrize('lang', RECALL_LANGUAGES)
def test_a_genuinely_untranslated_passage_is_caught_in_every_covered_language(lang):
    """The headline case the guard exists for. The French, English and German
    tables were always adequate; the other four were not, and their true
    positives sat at 0.36–0.38 — under the bar — because the lists were short
    rather than because the bar was wrong."""
    assert BookTranslator._is_in_source_language(SAMPLES[lang], lang), lang


@pytest.mark.parametrize('source,target', [
    (source, target)
    for source in SAMPLES for target in SAMPLES if source != target
])
def test_correct_text_in_any_language_is_never_read_as_another(source, target):
    """The guard refuses a replacement it believes is written in the source
    language, so a false positive here silently discards a good fix. This is
    the property that makes the six cognate directions — it↔es, es↔it,
    es↔pt, pt↔es, it↔pt, pt↔it — the ones worth being careful about, and the
    reason every marker added in the second pass had to be absent from every
    other table. All 56 ordered pairs, Korean included: nothing in this matrix
    fires."""
    assert not BookTranslator._is_in_source_language(SAMPLES[target], source), (
        f'{target} text flagged as {source}'
    )


def test_no_marker_appears_in_two_tables_except_the_cognates_already_shared():
    """The mechanical form of the rule the test above checks the consequence
    of. Every overlap here predates the second pass and is deliberate: these
    words are genuinely common to both languages, and a marker shared with a
    language this book might be translated into fires on good text there. The
    test exists because the second pass briefly broke it — `entre`, `nunca`
    and `sobre` were added to both Spanish and Portuguese, and `sempre` to both
    Italian and Portuguese, having been checked against the original tables
    but not against each other."""
    overlaps = set()
    for lang, words in BookTranslator._FUNCTION_WORDS.items():
        for other, other_words in BookTranslator._FUNCTION_WORDS.items():
            if other <= lang:
                continue
            shared = words & other_words
            if shared:
                overlaps.add(f'{lang}/{other}: {" ".join(sorted(shared))}')
    assert overlaps == {
        'de/en: in was',
        'de/it: in',
        'en/es: no',
        'en/fr: a',
        'en/it: come in',
        'en/pt: as do',
        'es/fr: de en la que son un',
        'es/pt: era estar mas o ser',
        'fr/it: il le',
        'fr/pt: mais ou',
        'it/pt: disse e quando',
    }, overlaps


def test_the_english_and_french_tables_are_the_ones_an_english_to_french_run_reads():
    """The dict is keyed by *source* language and read one way only, so an
    English→French run touches `en` alone and nothing added to another table
    can reach it. Pinned because that is what makes widening the other lists a
    safe change rather than a risk to the most-used pair."""
    assert BookTranslator._is_in_source_language(SAMPLES['en'], 'en')
    assert not BookTranslator._is_in_source_language(SAMPLES['fr'], 'en')
    # A run where source and target are the same language stands the guard down
    # before it ever reads a table, so these sizes only matter one direction.
    assert len(BookTranslator._FUNCTION_WORDS['en']) > 100
    assert len(BookTranslator._FUNCTION_WORDS['fr']) > 35


def test_the_language_guard_speaks_the_language_asked_for():
    """A French sentence is not English, and an English one is not French —
    the guard is not a generic 'is this foreign' test."""
    assert BookTranslator._is_in_source_language('Where are you from now?', 'en')
    assert not BookTranslator._is_in_source_language('Where are you from now?', 'fr')
    assert BookTranslator._is_in_source_language("D'où venez-vous maintenant ?", 'fr')
    assert not BookTranslator._is_in_source_language("D'où venez-vous maintenant ?", 'en')


def test_an_unknown_source_language_stands_the_guard_down():
    """A code the table does not cover must not start dropping legitimate
    fixes — a guard that cannot tell must keep quiet."""
    assert not BookTranslator._is_in_source_language('Where are you from?', 'xx')
    assert not BookTranslator._is_in_source_language('Where are you from?', '')


def test_the_loss_checks_run_before_a_span_is_counted_as_actionable(monkeypatch):
    """Order matters: a dropped error must not appear in the review desk as
    something a human is being asked to approve."""
    translator = BookTranslator(model_name='reviewer:12b')
    reported = json.dumps([
        {'span': 'Il aurait été préférable qu’elle soit en colère contre moi.',
         'replacement': 'It would have been better if she was mad at me.',
         'type': 'omission', 'severity': 'major'},
    ])
    _script_model_calls(monkeypatch, [reported])

    _, _, details = translator.stage2_reflection_improvement(
        original_text=FR_PAGE_SOURCE, draft_translation=FR_PAGE_DRAFT,
        source_lang='english', target_lang='french',
    )

    assert details['issues'] == []
    assert details['actionable_issues'] == []


# -- A passage left in the source language.
#
# The other bug this pass is here for: a paragraph that Stage 1 skipped, that
# Stage 2 then read as correct because it matched the source it was being
# checked against, and that reached the book in English.

EN_SOURCE = (
    'It would have been better if she was mad at me. She had not met her '
    'sister for several years, and he had become a very small man indeed.'
)


def test_a_passage_left_in_english_is_named_to_the_reviewer(monkeypatch):
    """The reviewer is asked to find errors in a translation, and an
    untranslated paragraph does not look like one when the source is printed
    above it. The pass can see it without a model, so it says so — the
    reviewer is told what to look at, never what to write."""
    calls = _script_model_calls(monkeypatch, ['[]'])
    BookTranslator(model_name='reviewer:12b').stage2_estimate(
        EN_SOURCE, EN_SOURCE, 'en', 'fr',
    )

    assert 'It is written in English' in calls[0]['prompt']


def test_a_translated_chunk_is_not_told_it_is_still_in_the_source_language(monkeypatch):
    calls = _script_model_calls(monkeypatch, ['[]'])
    BookTranslator(model_name='reviewer:12b').stage2_estimate(
        FR_PAGE_SOURCE, FR_PAGE_DRAFT, 'en', 'fr',
    )

    assert 'It is written in English' not in calls[0]['prompt']


@pytest.mark.parametrize('severity', ['critical', 'major', 'minor'])
def test_an_untranslated_passage_is_applied_whatever_severity_it_was_given(severity):
    """Whether a passage is translated is not a matter of degree, and the
    reviewer that calls a skipped paragraph "minor" is right about the length
    and wrong about the problem. It still has to be fixed."""
    assert BookTranslator.is_actionable_error(
        {'type': 'untranslated', 'severity': severity, 'span': 'x', 'replacement': 'y'}
    )


def test_the_aliases_the_models_produce_all_name_the_same_category():
    for alias in ('untranslated', 'left untranslated', 'untranslated text',
                  'not translated', 'source language'):
        assert BookTranslator.ERROR_TYPE_ALIASES.get(alias, alias) == 'untranslated'

    errors = BookTranslator.validate_estimate_spans(
        [{'span': 'Elle est là.', 'type': 'left untranslated', 'severity': 'minor',
          'replacement': 'Elle est toujours là.'}],
        'Elle est là.',
    )

    assert [error['type'] for error in errors] == ['untranslated']


def test_an_untranslated_passage_still_faces_the_verifier(monkeypatch):
    """Applying it at any severity must not make it a fact the pipeline trusts
    on its label: restoring a whole paragraph is a rewrite, and a rewrite goes
    to the vote like any other."""
    translator = BookTranslator(model_name='reviewer:12b', verifier_model='verifier:27b')
    reported = json.dumps([
        {'span': 'She had not met her sister for several years,',
         'type': 'untranslated', 'severity': 'minor',
         'replacement': 'Elle n’avait pas vu sa sœur depuis des années,'},
    ])

    text, _, details, calls = _refine_with_source(
        translator, monkeypatch, EN_SOURCE, reported, 'A', 'B',
    )

    assert 'Elle n’avait pas vu sa sœur' in text
    assert details['verified']['accepted'] is True
    assert [call['model'] for call in calls][1:] == ['verifier:27b', 'verifier:27b']


def test_the_same_language_pair_stands_the_source_guards_down(monkeypatch):
    """With one language there is no 'wrong language' to catch: every correct
    replacement matches by definition, so a guard left running would refuse
    every fix in the pass."""
    reported = json.dumps([
        {'span': 'Il aurait été préférable qu’elle soit en colère contre moi.',
         'type': 'omission', 'severity': 'major',
         'replacement': 'It would have been better if she was mad at me.'},
    ])
    calls = _script_model_calls(monkeypatch, [reported])

    errors, warning = BookTranslator(model_name='reviewer:12b').stage2_estimate(
        FR_PAGE_SOURCE, FR_PAGE_DRAFT, 'english', 'english',
    )

    assert warning is None
    assert [error['replacement'] for error in errors] == [
        'It would have been better if she was mad at me.',
    ]
    assert len(calls) == 1


def test_the_length_guard_leaves_a_correct_chinese_rendering_alone():
    """Chinese counts its length differently from every Latin language, and a
    correct translation of a paragraph comes out well under 40% of its
    characters. Judged by the Latin ratio it reads as a deletion — which is how
    the fix that puts the missing text back would be the one thing refused."""
    span = 'She had not met her sister for several years, and he had become a very small man indeed.'
    replacement = '她已经很多年没有见过姐姐了，他确实变得非常渺小。'

    assert len(replacement) < len(span) * BookTranslator.DELETED_TEXT_MIN_RATIO
    assert not BookTranslator._shrinks_to_nothing(span, replacement)


def test_the_length_guard_still_refuses_a_collapse_in_another_script():
    assert BookTranslator._shrinks_to_nothing(
        'She had not met her sister for several years, and he had become a very small man indeed.',
        'Super.',
    )


def test_a_refused_edit_is_counted_per_guard(monkeypatch):
    """A chunk whose whole review answer was refused reads as "0 found" in the
    log, and that is the reading that hides an untranslated paragraph."""
    translator = BookTranslator(model_name='reviewer:12b')
    reported = json.dumps([
        {'span': 'Il aurait été préférable qu’elle soit en colère contre moi.',
         'replacement': 'It would have been better if she was mad at me.',
         'type': 'omission', 'severity': 'major'},
    ])
    _script_model_calls(monkeypatch, [reported])

    _, _, details = translator.stage2_reflection_improvement(
        original_text=FR_PAGE_SOURCE, draft_translation=FR_PAGE_DRAFT,
        source_lang='english', target_lang='french',
    )

    assert details['errors_found'] == 0
    assert details['errors_dropped'] == 1
    assert details['dropped_by_guard'] == {'source copy': 1}
    assert '1 not applied (source copy 1)' in (
        BookTranslator._describe_stage2_chunk(details, warning=None, changed=False)
    )


def test_a_span_that_is_not_in_the_draft_is_counted_too(monkeypatch):
    """The commonest way a review answer yields nothing is a model reporting an
    error about a string that is not there. Uncounted, the chunk reads as
    "0 found · nothing to do", which is the reading that hides an untranslated
    paragraph — so it is counted, and named for what it is rather than for a
    guard that never ran on it."""
    translator = BookTranslator(model_name='reviewer:12b')
    reported = json.dumps([
        {'span': 'Elle n\'est jamais venue a la maison ce soir-la.',
         'replacement': 'Elle n\'est jamais venue à la maison ce soir-là.',
         'type': 'mistranslation', 'severity': 'major'},
        {'span': 'The dog barked all afternoon.',
         'replacement': 'Le chien a aboyé tout l\'après-midi.',
         'type': 'mistranslation', 'severity': 'major'},
    ])
    _script_model_calls(monkeypatch, [reported])

    _, _, details = translator.stage2_reflection_improvement(
        original_text=FR_PAGE_SOURCE, draft_translation=FR_PAGE_DRAFT,
        source_lang='english', target_lang='french',
    )

    assert details['errors_found'] == 0
    assert details['errors_dropped'] == 2
    assert details['dropped_by_guard'] == {'span not in draft': 2}
    assert '2 not applied (span not in draft 2)' in (
        BookTranslator._describe_stage2_chunk(details, warning=None, changed=False)
    )


def test_a_span_refused_by_a_guard_twice_is_counted_twice(monkeypatch):
    """Pins the upper bound documented on the chunk summary. The guards run
    before `seen`, so a duplicated answer takes the same branch twice and
    `errors_dropped` over-counts distinct problems. The alternative — marking
    the span seen even when a guard refuses it — would also stop a *second,
    different* proposal for the same span from being considered, which is a
    behaviour change and not a reporting fix. Counting twice is the safe
    direction: it never under-reports."""
    translator = BookTranslator(model_name='reviewer:12b')
    one = {'span': 'Il aurait été préférable qu’elle soit en colère contre moi.',
           'replacement': 'It would have been better if she was mad at me.',
           'type': 'omission', 'severity': 'major'}
    _script_model_calls(monkeypatch, [json.dumps([one, dict(one)])])

    _, _, details = translator.stage2_reflection_improvement(
        original_text=FR_PAGE_SOURCE, draft_translation=FR_PAGE_DRAFT,
        source_lang='english', target_lang='french',
    )

    assert details['errors_found'] == 0
    assert details['dropped_by_guard'] == {'source copy': 2}


def test_a_span_reported_twice_that_survives_is_counted_once(monkeypatch):
    """The `seen` discard is the one deliberately left uncounted: the first
    report of the span is already in `validated`, so counting the second would
    double-count one model mistake."""
    translator = BookTranslator(model_name='reviewer:12b')
    one = {'span': 'cette froideur, ce vide',
           'replacement': 'cette froideur absolute, ce vide total',
           'type': 'style', 'severity': 'major'}
    _script_model_calls(monkeypatch, [json.dumps([one, dict(one)])])

    _, _, details = translator.stage2_reflection_improvement(
        original_text=FR_PAGE_SOURCE, draft_translation=FR_PAGE_DRAFT,
        source_lang='english', target_lang='french',
    )

    assert details['errors_found'] == 1
    assert details['errors_dropped'] == 0


def _refine_with_source(translator, monkeypatch, source, reported, *answers):
    """Refine a chunk whose draft is the source text, i.e. Stage 1 skipped it."""
    calls = _script_model_calls(monkeypatch, [reported, *answers])
    text, warning, details = translator.stage2_reflection_improvement(
        original_text=source, draft_translation=source,
        source_lang='english', target_lang='french',
    )
    return text, warning, details, calls


# -- The verify half. Still no Ollama: the model answers are scripted, which
# is the only way to pin down who was asked and what was done with the reply.

SOURCE = 'Mr Dursley was the director of a firm called Grunnings, which made drills.'


def _script_model_calls(monkeypatch, answers):
    """Replace every model call with a scripted answer, recording the model."""
    calls = []

    def fake_call(self, prompt, temperature=0.2, read_timeout=180):
        calls.append({'model': self.model_name, 'prompt': prompt})
        return answers.pop(0) if answers else None

    monkeypatch.setattr(BookTranslator, '_call_model', fake_call)
    return calls


def _refine(translator, monkeypatch, answers):
    calls = _script_model_calls(monkeypatch, answers)
    text, warning, details = translator.stage2_reflection_improvement(
        original_text=SOURCE,
        draft_translation=DRAFT,
        source_lang='english',
        target_lang='russian',
    )
    return text, warning, details, calls


def _reported(error_type, span='свёрла', replacement='дрели', severity='major'):
    return (
        '[{"span": "%s", "type": "%s", "severity": "%s", "replacement": "%s"}]'
        % (span, error_type, severity, replacement)
    )


def test_the_verdict_is_asked_of_the_verifier_model_not_the_reviewer(monkeypatch):
    """The reviewer grading its own edit is the arrangement that made this
    pass a no-op, so the two calls must reach the other model."""
    translator = BookTranslator(model_name='reviewer:12b', verifier_model='verifier:27b')

    # 'A' then 'B' is the patched version winning both orderings.
    text, warning, details, calls = _refine(
        translator, monkeypatch, [_reported('mistranslation'), 'A', 'B'],
    )

    assert [call['model'] for call in calls] == ['reviewer:12b', 'verifier:27b', 'verifier:27b']
    assert text == 'Мистер Дурсли был директором фирмы Grunnings, которая делала дрели.'
    assert warning is None
    assert details['verified'] == {
        'verdicts': ['patched', 'patched'],
        'accepted': True,
        'model': 'verifier:27b',
    }


def test_position_bias_is_named_and_neutral_rejection_keeps_the_draft(monkeypatch):
    """Answering 'A' both times is position bias, not agreement — and it is
    what a model asked to grade its own edit does. The draft is kept, and the
    details say who said so."""
    translator = BookTranslator(model_name='reviewer:12b')

    text, _, details, calls = _refine(
        translator, monkeypatch, [_reported('mistranslation'), 'A', 'A', 'REJECT'],
    )

    assert text == DRAFT
    assert details['verified']['verdicts'] == ['patched', 'draft']
    assert details['verified']['position_bias_detected'] is True
    assert details['verified']['neutral_check'] == 'rejected'
    assert details['verified']['accepted'] is False
    # No separate verifier chosen: every call went to the reviewing model.
    assert {call['model'] for call in calls} == {'reviewer:12b'}


def test_position_biased_vote_can_be_resolved_without_ordered_versions(monkeypatch):
    translator = BookTranslator(model_name='reviewer:12b', verifier_model='verifier:27b')

    text, _, details, calls = _refine(
        translator, monkeypatch, [_reported('mistranslation'), 'A', 'A', 'ACCEPT'],
    )

    assert text.endswith('которая делала дрели.')
    assert details['verified']['position_bias_detected'] is True
    assert details['verified']['neutral_check'] == 'accepted'
    assert details['verified']['accepted'] is True
    assert 'VERSION A' not in calls[-1]['prompt']
    assert '"свёрла" => "дрели"' in calls[-1]['prompt']


def test_tie_uses_the_neutral_edit_check_instead_of_becoming_an_automatic_veto(
    monkeypatch,
):
    translator = BookTranslator(model_name='reviewer:12b', verifier_model='verifier:27b')

    text, _, details, _ = _refine(
        translator, monkeypatch, [_reported('mistranslation'), 'TIE', 'TIE', 'ACCEPT'],
    )

    assert text != DRAFT
    assert details['verified']['tie_detected'] is True
    assert details['verified']['neutral_check'] == 'accepted'


def test_draft_winning_both_orders_is_rejected_without_a_fallback_call(monkeypatch):
    translator = BookTranslator(model_name='reviewer:12b', verifier_model='verifier:27b')

    text, _, details, calls = _refine(
        translator, monkeypatch, [_reported('mistranslation'), 'B', 'A'],
    )

    assert text == DRAFT
    assert details['verified']['verdicts'] == ['draft', 'draft']
    assert details['verified']['accepted'] is False
    assert 'neutral_check' not in details['verified']
    assert len(calls) == 3


def test_a_verifier_that_never_answers_is_recorded_as_unavailable(monkeypatch):
    translator = BookTranslator(model_name='reviewer:12b', verifier_model='verifier:27b')

    text, _, details, _ = _refine(
        translator, monkeypatch, [_reported('mistranslation')],  # verifier: no answer
    )

    assert text == DRAFT
    assert details['verified']['verdicts'] == ['unavailable', 'unavailable']


@pytest.mark.parametrize('error_type', ['omission', 'addition', 'terminology', 'consistency'])
def test_fixes_checkable_against_the_source_skip_the_verifier(monkeypatch, error_type):
    """Whether a clause is missing, invented, or rendered against the
    glossary is settled by reading the two texts. The A/B vote adds nothing
    there and can only veto a real fix."""
    translator = BookTranslator(model_name='reviewer:12b', verifier_model='verifier:27b')

    text, _, details, calls = _refine(
        translator, monkeypatch, [_reported(error_type)],
    )

    assert text != DRAFT
    assert details['verified'] == 'skipped_objective'
    assert len(calls) == 1  # estimate only — the verifier was never asked


def test_an_objective_category_covering_the_page_is_still_put_to_the_vote(monkeypatch):
    """The exemption is for fixes small enough to be a fact about the source.
    Reported as omissions, a run of spans that between them cover the draft is
    a rewrite wearing that label, and it goes to the verifier like any other.

    This is the shape of the observed failure: a run of `omission` errors,
    each individually exempt, together replacing most of the chunk. The
    replacements here are rewrites rather than copies of the source, so this
    case is about the vote and not about the copy guard."""
    translator = BookTranslator(model_name='reviewer:12b', verifier_model='verifier:27b')
    reported = json.dumps([
        {'span': 'Мистер Дурсли был директором', 'type': 'omission',
         'severity': 'major', 'replacement': 'Г-н Дурсли управлял конторой'},
        {'span': ' фирмы Grunnings, которая делала свёрла.',
         'type': 'omission', 'severity': 'major',
         'replacement': ' под названием Grunnings, выпускавшей свёрла.'},
    ])

    text, _, details, calls = _refine(translator, monkeypatch, [reported, 'A', 'B'])

    assert details['verified']['accepted'] is True
    assert [call['model'] for call in calls] == ['reviewer:12b', 'verifier:27b', 'verifier:27b']


def test_a_vetoed_rewrite_keeps_the_draft_rather_than_half_of_it(monkeypatch):
    """If the vote goes the other way the whole patch is dropped, so a rewrite
    that turns out to be wrong cannot leave the chunk in pieces."""
    translator = BookTranslator(model_name='reviewer:12b', verifier_model='verifier:27b')
    reported = json.dumps([
        {'span': 'Мистер Дурсли был директором', 'type': 'omission',
         'severity': 'major', 'replacement': 'Г-н Дурсли управлял конторой'},
        {'span': ' фирмы Grunnings, которая делала свёрла.',
         'type': 'omission', 'severity': 'major',
         'replacement': ' под названием Grunnings, выпускавшей свёрла.'},
    ])

    text, _, details, _ = _refine(translator, monkeypatch, [reported, 'B', 'A'])

    assert text == DRAFT
    assert details['verified']['accepted'] is False


def test_a_span_covering_its_whole_chunk_counts_as_a_rewrite():
    """The share is measured against the draft, so in a very short chunk even
    one small span is the whole text — and replacing all of it deserves the
    vote, not a category label."""
    applied = [{'span': 'Super.', 'type': 'omission', 'replacement': 'Great.'}]

    assert BookTranslator._is_rewrite('Super.', applied) is True
    assert BookTranslator._is_rewrite(
        'Super. J\'étais devant sa porte et je ne pouvais plus franchir son seuil.',
        applied,
    ) is False


def test_a_long_draft_keeps_a_small_objective_fix_off_the_vote():
    """The common case must not get slower: a glossary term corrected in a
    full page of prose is a small share of that page."""
    draft = 'Il est resté là. ' * 20
    applied = [{'span': 'Il est', 'type': 'terminology', 'replacement': 'Il était'}]

    assert BookTranslator._is_rewrite(draft, applied) is False


def test_a_mixed_patch_still_faces_the_verifier(monkeypatch):
    """One mistranslation among the objective fixes puts the whole patch back
    under the vote — the spans are applied together and cannot be split."""
    translator = BookTranslator(model_name='reviewer:12b', verifier_model='verifier:27b')

    reported = (
        '[{"span": "свёрла", "type": "omission", "severity": "major", "replacement": "дрели"},'
        ' {"span": "Grunnings", "type": "mistranslation", "severity": "major",'
        ' "replacement": "Граннингс"}]'
    )
    _, _, details, calls = _refine(translator, monkeypatch, [reported, 'A', 'B'])

    assert details['verified']['accepted'] is True
    assert [call['model'] for call in calls] == ['reviewer:12b', 'verifier:27b', 'verifier:27b']


def test_no_separate_verifier_builds_no_second_translator():
    translator = BookTranslator(model_name='reviewer:12b')

    assert translator.verifier is translator
    assert translator.verifier_model == 'reviewer:12b'


def test_details_name_both_models_so_a_run_can_be_read_back(monkeypatch):
    translator = BookTranslator(model_name='reviewer:12b', verifier_model='verifier:27b')

    _, _, details, _ = _refine(translator, monkeypatch, ['[]'])

    assert details['review_model'] == 'reviewer:12b'
    assert details['verifier_model'] == 'verifier:27b'


def test_the_chunk_log_line_distinguishes_the_three_ways_nothing_changes():
    clean = BookTranslator._describe_stage2_chunk(
        {'errors_found': 0, 'errors_actionable': 0, 'errors_applied': 0, 'verified': None},
        warning=None, changed=False,
    )
    vetoed = BookTranslator._describe_stage2_chunk(
        {
            'errors_found': 2, 'errors_actionable': 2, 'errors_applied': 2,
            'verified': {
                'verdicts': ['patched', 'draft'],
                'accepted': False,
                'model': 'verifier:27b',
                'position_bias_detected': True,
                'neutral_check': 'rejected',
            },
        },
        warning=None, changed=False,
    )
    timed_out = BookTranslator._describe_stage2_chunk(
        {'errors_found': 0, 'errors_actionable': 0, 'errors_applied': 0, 'verified': None},
        warning='The review pass returned no output — kept the draft for this chunk.',
        changed=False,
    )

    assert 'verifier not needed' in clean
    assert 'verifier verifier:27b voted patched/draft' in vetoed
    assert 'position bias detected' in vetoed
    assert 'neutral edit check rejected → rejected' in vetoed
    assert 'review pass gave no answer' in timed_out
    assert clean != timed_out


def test_stage2_cache_key_changes_when_only_the_verifier_changes():
    first = BookTranslator(
        model_name='reviewer:12b', verifier_model='verifier-a:27b',
    )._stage2_cache_model('same-glossary')
    second = BookTranslator(
        model_name='reviewer:12b', verifier_model='verifier-b:27b',
    )._stage2_cache_model('same-glossary')

    assert first != second
    assert 'verifier-a:27b' in first
    assert 'verifier-b:27b' in second
