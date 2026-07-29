"""The document-level Quality Check tests.

These are the ones that can see what a chunk-level judge cannot: a name
rendered here and dropped there, a word left in the source's script, a target
text that came out suspiciously the same size as its source.
"""

from translator import BookTranslator, TerminologyManager


translator = BookTranslator('qwen3:4b-instruct')


ORIGINAL_CHUNKS = [
    'Mr. Dursley was the director of a firm called Grunnings, which made drills.',
    'The Dursleys had a small son called Dudley.',
    'Mr. Dursley left the house and drove towards London.',
]


def test_script_leakage_finds_an_untranslated_name():
    result = translator.eval_script_leakage(
        ' '.join(ORIGINAL_CHUNKS),
        'Мистер Дурсль был директором фирмы Grunnings, которая делала дрели.',
    )

    assert result['flagged']
    assert result['value'] == 1
    assert result['details']['words'] == {'Grunnings': 1}
    assert result['details']['target_script'] == 'Cyrillic'


def test_script_leakage_stays_quiet_on_a_clean_translation():
    result = translator.eval_script_leakage(
        ' '.join(ORIGINAL_CHUNKS),
        'Мистер Дурсль был директором фирмы Граннингс, которая делала дрели.',
    )

    assert not result['flagged']
    assert result['value'] == 0


def test_script_leakage_declines_to_guess_within_one_script():
    """English to German shares an alphabet, so an untranslated word is
    indistinguishable from a translated one this way. Saying so is better
    than reporting every word as suspicious or none as."""
    result = translator.eval_script_leakage(
        'The garden was quiet.', 'Der Garten war still.',
    )

    assert not result['flagged']
    assert 'same script' in result['note']


def test_numeric_preservation_catches_lost_date_and_quantity():
    result = translator.eval_numeric_preservation(
        'On 31/10/1991, she bought 2 tickets for 1,500 dollars.',
        '31/10/1991 она купила два билета за 1500 долларов.',
    )

    assert result['flagged']
    assert result['details']['missing'] == ['2']
    # Thousands separators are normalized, rather than producing a false alarm.
    assert '1500' not in result['details']['missing']


def test_numeric_preservation_accepts_identical_digit_facts():
    result = translator.eval_numeric_preservation(
        'Chapter 4 starts at 09:30 on 12.05.2024.',
        'Глава 4 начинается в 09:30 12.05.2024.',
    )

    assert not result['flagged']
    assert result['value'] == 0


def test_chunk_coverage_detects_missing_final_chunk():
    result = translator.eval_chunk_coverage(['one', 'two', 'three'], ['один', '', 'три'])

    assert result['flagged']
    assert result['details']['empty_final_chunks'] == [2]


def test_chunk_coverage_detects_changed_chunk_count():
    result = translator.eval_chunk_coverage(['one', 'two'], ['один'])

    assert result['flagged']
    assert result['details']['source_chunks'] == 2
    assert result['details']['final_chunks'] == 1


def test_entity_consistency_reports_the_chunks_a_name_is_missing_from():
    terminology = TerminologyManager.from_text('Dursley => Дурсль | inflectable')
    final_chunks = [
        'Мистер Дурсль был директором фирмы Граннингс, которая делала дрели.',
        'У Дурслей был маленький сын по имени Дадли.',
        # The name is simply gone from this one.
        'Он вышел из дома и поехал в Лондон.',
    ]

    result = translator.eval_entity_consistency(ORIGINAL_CHUNKS, final_chunks, terminology)

    assert result['flagged']
    finding = result['details']['findings'][0]
    assert finding['source'] == 'Dursley'
    assert finding['chunks_with_source'] == 3
    assert finding['chunks_with_rendering'] == 2


def test_entity_consistency_accepts_inflected_forms():
    """Дурсль/Дурсля/Дурслю are one name doing its job in a case language,
    not three renderings — pinning the name to one surface form would make
    the surrounding sentence ungrammatical."""
    terminology = TerminologyManager.from_text('Dursley => Дурсль | inflectable')
    final_chunks = [
        'Мистер Дурсль был директором фирмы Граннингс.',
        'У Дурсля был маленький сын по имени Дадли.',
        'Дурслю пришлось выйти из дома и поехать в Лондон.',
    ]

    result = translator.eval_entity_consistency(ORIGINAL_CHUNKS, final_chunks, terminology)

    assert not result['flagged']
    assert result['value'] == 0


def test_entity_consistency_rejects_a_shortened_name_that_shares_a_stem():
    terminology = TerminologyManager.from_text('Fenwick => Фенвикс | inflectable')
    result = translator.eval_entity_consistency(
        ['Mr. Fenwick went home.'], ['Мистер Фенвик пошёл домой.'], terminology,
    )

    assert result['flagged']
    assert result['details']['findings'][0]['source'] == 'Fenwick'


def test_entity_consistency_admits_which_names_it_cannot_tell_apart():
    """Дурсль and Дурсли differ only in the ending that inflection is
    allowed to change, so stem matching cannot police them. Saying which
    pairs those are beats reporting them as fine."""
    terminology = TerminologyManager.from_text(
        'Dursley => Дурсль | inflectable\nDursleys => Дурсли | inflectable'
    )

    result = translator.eval_entity_consistency(
        ['Mr. Dursley drove to work.', 'The Dursleys had a son.'],
        ['Мистер Дурсль поехал на работу.', 'У Дурслей был сын.'],
        terminology,
    )

    assert result['details']['ambiguous_pairs'] == [['Dursley', 'Dursleys']]
    assert 'check them by hand' in result['note']


def test_entity_consistency_accepts_an_inflected_short_name():
    """A four-letter vowel-final name loses that vowel to the ending, so
    holding it to its full form reported every correct oblique form as the
    name being missing."""
    terminology = TerminologyManager.from_text('The Water => Река | inflectable')
    result = translator.eval_entity_consistency(
        ['The Hill stood over The Water.'],
        ['Холм возвышался над рекой у реки.'],
        terminology,
    )

    assert not result['flagged']
    assert result['value'] == 0
    assert result['details']['loosely_matched'] == ['Река']
    assert 'shortened stem' in result['note']


def test_entity_consistency_does_not_let_a_longer_word_stand_in_for_a_short_name():
    """The shortened stem of "Холм" is a prefix of "холодный" too. Endings
    are short, so length is what separates the two."""
    terminology = TerminologyManager.from_text('The Hill => Холм | inflectable')
    result = translator.eval_entity_consistency(
        ['The Hill was quiet.'], ['Холодный ветер стих.'], terminology,
    )

    assert result['flagged']
    assert result['details']['findings'][0]['source'] == 'The Hill'


def test_terminology_delta_says_when_the_glossary_gives_it_nothing_to_enforce():
    """Only exact terms can be checked by literal match, so a glossary of
    inflectable names produces a green this test cannot fail — which reads
    exactly like a clean run and is not one."""
    terminology = TerminologyManager.from_text(
        'Dursley => Дурсль | inflectable\nDudley => Дадли | preferred'
    )

    result = translator.eval_terminology_delta(
        'Mr. Dursley had a son called Dudley.',
        'Мистер Д. и его сын.',
        'Мистер Д. и его сын.',
        terminology,
    )

    assert not result['flagged']
    assert 'Nothing to enforce' in result['note']
    assert 'Named-entity consistency' in result['note']


def test_terminology_delta_still_counts_exact_violations():
    terminology = TerminologyManager.from_text('Dursley => Дурсль | exact')

    result = translator.eval_terminology_delta(
        'Mr. Dursley drove to work.',
        'Мистер Дурслей поехал на работу.',
        'Мистер Дурсль поехал на работу.',
        terminology,
    )

    assert not result['flagged']
    assert result['value'] == 1
    assert 'fixed 1 glossary violation' in result['note']


def test_entity_consistency_says_so_instead_of_passing_on_an_empty_glossary():
    """An empty glossary made this check pass unconditionally, which for a
    text whose main risk is proper nouns is the most expensive kind of
    green."""
    result = translator.eval_entity_consistency(
        ORIGINAL_CHUNKS, ['x', 'y', 'z'], TerminologyManager(),
    )

    assert result['flagged']
    assert result['value'] is None
    assert 'Run Prepare' in result['note']


def test_length_ratio_is_measured_against_the_source():
    source = 'a' * 1000
    draft = 'б' * 1010
    final = 'б' * 1000

    result = translator.eval_length_ratio(source, draft, final)

    assert result['value'] == 1.0
    assert result['details']['source_chars'] == 1000
    # The old final/draft number is still reported, just no longer the headline.
    assert result['details']['final_over_draft'] == 0.99


def test_length_ratio_flags_a_truncated_translation():
    result = translator.eval_length_ratio('a' * 1000, 'б' * 400, 'б' * 400)

    assert result['flagged']
    assert result['value'] == 0.4


def test_risk_sampling_prefers_chunks_with_names_dialogue_and_numbers():
    chunks = [
        'The weather was mild and the hills were green and the road went on.',
        'The weather was mild and the hills were green and the road went on.',
        '"Get in, Dudley," said Mr. Dursley, "we leave for Grunnings at 8."',
        'The weather was mild and the hills were green and the road went on.',
    ]

    assert 2 in BookTranslator._risk_ranked_indices(chunks, 1)


def test_risk_sampling_falls_back_to_even_spacing_when_nothing_scores():
    chunks = ['plain prose here'] * 4

    assert BookTranslator._risk_ranked_indices(chunks, 2) == [0, 3]
