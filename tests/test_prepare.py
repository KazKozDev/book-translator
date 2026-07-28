"""Stage 0: harvesting the names a document has to stay consistent about.

No model is involved in any of this, which is the point — the expensive
model call only has to render a list it is handed, not find it.
"""

from translator import BookTranslator


CHAPTER = """Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say
that they were perfectly normal, thank you very much.

Mr. Dursley was the director of a firm called Grunnings, which made drills.
He was a big, beefy man with hardly any neck. Mrs. Dursley was thin and blonde
and had nearly twice the usual amount of neck. The Dursleys had a small son
called Dudley and in their opinion there was no finer boy anywhere.

The Dursleys had everything they wanted, but they also had a secret, and their
greatest fear was that somebody would discover it. They didn't think they could
bear it if anyone found out about the Potters. Mrs. Potter was Mrs. Dursley's
sister, but they hadn't met for several years.
"""


def harvested():
    return {
        record['surface']: record
        for record in BookTranslator.harvest_proper_noun_candidates(CHAPTER)
    }


def test_finds_the_recurring_names():
    surfaces = harvested()

    for name in ('Dursley', 'Dursleys', 'Grunnings', 'Privet Drive', 'Dudley', 'Potters'):
        assert name in surfaces, f'{name} was not harvested'


def test_singular_and_plural_are_separate_records():
    """The whole point of harvesting both: "Dursley" and "Dursleys" need
    different renderings, and a check that only knows about one of them
    cannot see the other drifting."""
    surfaces = harvested()

    assert 'Dursley' in surfaces and 'Dursleys' in surfaces
    assert surfaces['Dursley']['count'] > surfaces['Dursleys']['count']


def test_a_name_after_an_abbreviation_still_counts_as_mid_sentence():
    """"said Mr. Dursley" must not read as a sentence opener, or most
    mentions of most characters lose the evidence that they are names."""
    assert harvested()['Dursley']['mid_sentence'] > 0


def test_sentence_openers_and_possessives_are_not_names():
    surfaces = harvested()

    # "They didn't think…" opens a sentence and "they" is used lowercase
    # elsewhere in the same text.
    assert 'They' not in surfaces
    # "Mrs. Dursley's sister" is the same name, not a second one.
    assert "Dursley's" not in surfaces


def test_leading_article_is_dropped_from_a_name():
    assert 'The Dursleys' not in harvested()


def test_honorific_is_not_proposed_as_a_separate_name():
    candidates = BookTranslator.harvest_proper_noun_candidates(
        'Mr. Fenwick met Mrs. Fenwick on Willow Street.'
    )

    assert {record['surface'] for record in candidates} == {'Fenwick', 'Willow Street'}


def test_a_rendering_is_rejected_unless_its_source_is_a_literal_span():
    """The one grounding rule: the model may notice a name the extractors
    missed, and may not invent a character or quietly respell one."""
    text = 'Mr. Fenwick met Mrs. Fenwick on Willow Street.'

    # Not in the candidate list, but written exactly as the document has it.
    assert BookTranslator._rendering_record(
        text, {'source': 'Willow Street', 'target': 'Уиллоу-стрит'}, {},
    )['source'] == 'Willow Street'
    # Invented outright.
    assert BookTranslator._rendering_record(
        text, {'source': 'Brightwell', 'target': 'Брайтуэлл'}, {},
    ) is None
    # Respelled rather than copied.
    assert BookTranslator._rendering_record(
        text, {'source': 'Willow street', 'target': 'Уиллоу-стрит'}, {},
    ) is None
    # A bare honorific is not an entity.
    assert BookTranslator._rendering_record(
        text, {'source': 'Mrs.', 'target': 'миссис'}, {},
    ) is None


def test_the_model_chooses_the_enforcement_mode_within_the_allowed_set():
    text = 'The Nimbus 2000 was made by Grunnings.'

    assert BookTranslator._rendering_record(
        text, {'source': 'Nimbus 2000', 'target': 'Нимбус 2000', 'mode': 'exact'}, {},
    )['mode'] == 'exact'
    # An unusable answer falls back rather than producing an unparseable line.
    assert BookTranslator._rendering_record(
        text, {'source': 'Grunnings', 'target': 'Граннингс', 'mode': 'whatever'}, {},
    )['mode'] == 'inflectable'


def test_every_candidate_carries_its_own_context_into_the_prompt():
    """A single truncated excerpt showed the model an opening it had no names
    in; each candidate bringing its own quote is the point of batching."""
    translator = BookTranslator.__new__(BookTranslator)
    prompt = BookTranslator._rendering_prompt(
        translator,
        'x' * 50000,
        [{'surface': 'Grunnings', 'count': 4, 'kind': 'organisation',
          'evidence': ['a firm called Grunnings, which made drills'],
          'variants': [{'form': 'Grunnings'}, {'form': 'GRUNNINGS'}]}],
        'en', 'ru', 'fiction',
    )

    assert 'a firm called Grunnings, which made drills' in prompt
    assert 'also written: GRUNNINGS' in prompt
    assert 'x' * 1000 not in prompt


def test_clustering_guesses_are_collected_for_the_model_to_rule_on():
    groups = BookTranslator.cluster_review_groups(
        CHAPTER,
        [{'surface': 'Dursley', 'count': 8, 'kind': 'person', 'evidence': ['Mr. Dursley'],
          'variants': [{'form': 'Dursley'}, {'form': 'Mrs. Dursley'}]},
         {'surface': 'Dudley', 'count': 3, 'kind': 'person', 'evidence': [], 'variants': []}],
        [{'a': 'Potter', 'b': 'Potters', 'context_a': 'the Potters', 'context_b': 'Mrs. Potter'}],
    )

    # The auto-merged cluster and the unresolved pair both need a ruling; the
    # single-form candidate does not.
    assert [group['merged_by'] for group in groups] == ['embeddings', 'unresolved']
    assert groups[0]['forms'] == ['Dursley', 'Mrs. Dursley']
    assert groups[1]['forms'] == ['Potter', 'Potters']


def test_a_confirmed_group_becomes_one_entry_and_a_rejected_one_splits_back():
    """The review queue used to be counted and dropped. A rejected merge has
    to actually split, or the two entities keep sharing one rendering."""
    text = 'Mr. Dursley met Mrs. Dursley. The Dursleys lived nearby. Dursley frowned.'
    candidates = [
        {'surface': 'Dursley', 'count': 3, 'kind': 'person', 'evidence': ['Dursley frowned'],
         'variants': [{'form': 'Dursley'}, {'form': 'Mrs. Dursley'}]},
        {'surface': 'Dursleys', 'count': 1, 'kind': 'organisation', 'evidence': [], 'variants': []},
    ]

    applied = BookTranslator.apply_cluster_decisions(text, candidates, [
        {'forms': ['Dursley', 'Mrs. Dursley'], 'same_entity': True, 'canonical': 'Dursley'},
        {'forms': ['Dursley', 'Dursleys'], 'same_entity': False, 'canonical': 'Dursley'},
    ])
    surfaces = [record['surface'] for record in applied]

    assert 'Dursley' in surfaces and 'Dursleys' in surfaces
    assert len(surfaces) == len(set(surfaces))


def test_a_split_ruling_never_produces_a_form_the_document_lacks():
    text = 'Fenwick arrived.'

    applied = BookTranslator.apply_cluster_decisions(
        text,
        [{'surface': 'Fenwick', 'count': 1, 'kind': 'person', 'evidence': [], 'variants': []}],
        [{'forms': ['Fenwick', 'Brightwell'], 'same_entity': False, 'canonical': 'Fenwick'}],
    )

    assert [record['surface'] for record in applied] == ['Fenwick']


def test_honorific_aliases_collapse_but_plural_family_name_stays_distinct():
    text = 'Mr. Dursley met Mrs. Dursley. The Dursleys lived nearby.'
    collapsed = BookTranslator.collapse_honorific_aliases(text, [
        {'surface': 'Mr. Dursley', 'canonical_source': 'Mr. Dursley', 'kind': 'person', 'evidence': ['Mr. Dursley'], 'count': 1},
        {'surface': 'Mrs. Dursley', 'canonical_source': 'Mrs. Dursley', 'kind': 'person', 'evidence': ['Mrs. Dursley'], 'count': 1},
        {'surface': 'Dursleys', 'canonical_source': 'Dursleys', 'kind': 'family', 'evidence': ['The Dursleys'], 'count': 1},
    ])

    assert [entity['canonical_source'] for entity in collapsed] == ['Dursley', 'Dursleys']


def test_rendering_collision_is_reported_before_it_reaches_start():
    conflicts = BookTranslator.find_rendering_conflicts([
        {'source': 'Dursley', 'target': 'Дурслей'},
        {'source': 'Dursleys', 'target': 'Дурслей'},
        {'source': 'Dudley', 'target': 'Дадли'},
    ])

    assert conflicts == [{
        'target': 'дурслей',
        'sources': ['Dursley', 'Dursleys'],
        'reason': 'distinct source entities have the same target rendering',
    }]


def test_names_below_the_builders_frequency_floor_still_reach_the_model():
    """The glossary builder only reports a cluster once it recurs often
    enough, and the rendering call may only render sources the candidate list
    established. Without the harvest unioned in, a street and a firm named
    twice in a chapter are unrenderable and Stage 0 returns a near-empty
    glossary for a text full of names."""
    merged = {
        record['surface']: record
        for record in BookTranslator.merge_harvested_candidates(CHAPTER, [
            {'surface': 'Mrs. Dursley', 'count': 6, 'kind': 'person', 'evidence': [], 'variants': []},
            {'surface': 'Dursleys', 'count': 4, 'kind': 'organisation', 'evidence': [], 'variants': []},
        ])
    }

    for name in ('Privet Drive', 'Grunnings', 'Dudley', 'Potters'):
        assert name in merged, f'{name} never reached the rendering call'
    # The neural list contributes its type; the harvest only ever guesses.
    assert merged['Dursleys']['kind'] == 'organisation'


def test_a_titled_mention_does_not_become_a_second_entry_for_one_person():
    """The builder reports "Mrs. Dursley" and the harvest reports the bare
    surname. Two entries for one person is how two different renderings get
    agreed, which is the failure Stage 0 exists to prevent."""
    merged = BookTranslator.merge_harvested_candidates(CHAPTER, [
        {'surface': 'Mrs. Dursley', 'count': 6, 'kind': 'person', 'evidence': [], 'variants': []},
    ])
    surfaces = [record['surface'] for record in merged]

    assert 'Mrs. Dursley' not in surfaces
    assert 'Dursley' in surfaces
    assert surfaces.count('Dursley') == 1


def test_returns_nothing_for_a_script_without_case():
    assert BookTranslator.harvest_proper_noun_candidates(
        '人工知能の研究は東京で始まった。人工知能は難しい。'
    ) == []

def test_a_confirmed_group_is_entered_under_the_bare_name():
    """The model may nominate the titled mention as canonical. The entry still
    has to be the bare name, or the constraint matches almost none of the
    occurrences it was agreed for."""
    text = 'Mrs. Dursley was thin. Dursley frowned. Mrs. Dursley left.'

    applied = BookTranslator.apply_cluster_decisions(text, [
        {'surface': 'Mrs. Dursley', 'count': 2, 'kind': 'person', 'evidence': [], 'variants': []},
        {'surface': 'Dursley', 'count': 3, 'kind': 'person', 'evidence': [], 'variants': []},
    ], [{'forms': ['Dursley', 'Mrs. Dursley'], 'same_entity': True, 'canonical': 'Mrs. Dursley'}])

    assert [record['surface'] for record in applied] == ['Dursley']


def test_stripping_a_title_leaves_an_untitled_name_alone():
    text = 'Mrs. Dursley left. Dursley too. Privet Drive was quiet.'

    assert BookTranslator.strip_honorific(text, 'Mrs. Dursley') == 'Dursley'
    assert BookTranslator.strip_honorific(text, 'Mr. and Mrs. Dursley') == 'Dursley'
    assert BookTranslator.strip_honorific(text, 'Privet Drive') == 'Privet Drive'


def test_a_titled_compound_is_not_accepted_as_an_entry():
    """"Mr. and Mrs. Dursley" is a real span of the document and the wrong
    entry: its rendering would match none of the plain mentions."""
    text = 'Mr. and Mrs. Dursley, of number four, Privet Drive. Dursley frowned.'

    assert BookTranslator._rendering_record(
        text, {'source': 'Mr. and Mrs. Dursley', 'target': 'мистер и миссис Дурсль'}, {},
    ) is None
    assert BookTranslator._rendering_record(
        text, {'source': 'Dursley', 'target': 'Дурсль'}, {},
    )['source'] == 'Dursley'


def test_a_harvested_candidate_arrives_with_context_of_its_own():
    """Without a quote, "Grunnings" reached the model as a bare token seen
    once — indistinguishable from a fragment of a chapter heading, and skipped
    along with it."""
    merged = {
        record['surface']: record
        for record in BookTranslator.merge_harvested_candidates(CHAPTER, [])
    }

    assert 'a firm called Grunnings' in ' '.join(merged['Grunnings']['evidence'])
    assert all(record['evidence'] for record in merged.values())


ESSAY = """Walt Whitman has somewhere a fine and just distinction. And in the sect
of Austenians or Janites, there would be found partisans. To some the freshness of
Northanger Abbey obscures the critical facts. Persuasion, relatively faint in tone,
has devotees. The catastrophe of Mansfield Park is theatrical, and Edmund only took
Fanny because Mary shocked him. Although Miss Austen liked the misunderstanding kind,
she was satisfied here. Miss Austen was barely twenty-one. A fondness for Miss Austen
is a patent of exemption. The transactions between Frank Churchill and Jane Fairfax
contribute to the intrigue. Jane Austen's genius had nothing mannish in it.
"""


def test_equal_counts_are_ordered_by_first_appearance_not_by_spelling():
    """Every term named once ties on frequency, and the tie used to break on
    the surface string. A budget that then cuts the list ran out mid-alphabet,
    so which names a reader saw depended on their initials."""
    once = [
        record for record in BookTranslator.merge_harvested_candidates(ESSAY, [])
        if record['count'] == 1
    ]
    positions = [ESSAY.find(record['surface']) for record in once]

    assert positions == sorted(positions), [record['surface'] for record in once]
    # Not a tautology: the two orders genuinely disagree on this text.
    assert [record['surface'] for record in once] != sorted(
        record['surface'] for record in once
    )


def test_a_name_that_only_ever_opens_a_sentence_is_still_harvested():
    """"Persuasion" and "Walt Whitman" are each named once, each at the head
    of a sentence. Requiring three mentions lost both."""
    surfaces = {record['surface'] for record in BookTranslator.merge_harvested_candidates(ESSAY, [])}

    assert 'Persuasion' in surfaces
    assert 'Walt Whitman' in surfaces
    assert 'Northanger Abbey' in surfaces
    assert 'Mansfield Park' in surfaces


def test_a_sentence_opener_is_not_glued_to_the_name_behind_it():
    """The document decides, not a list of function words: the tail of
    "Although Miss Austen" is a candidate on its own, the tail of "Walt
    Whitman" is not."""
    surfaces = {record['surface'] for record in BookTranslator.merge_harvested_candidates(ESSAY, [])}

    assert 'Although Miss Austen' not in surfaces
    assert 'Walt Whitman' in surfaces


def test_a_bare_given_name_shared_by_two_people_is_not_pinned():
    """One glossary line is one rendering for every occurrence of its source,
    so "Jane" cannot serve Jane Austen and Jane Fairfax at once."""
    surfaces = {record['surface'] for record in BookTranslator.merge_harvested_candidates(ESSAY, [])}

    assert 'Jane' not in surfaces
    assert 'Jane Austen' in surfaces and 'Jane Fairfax' in surfaces


def test_a_pronoun_the_neural_extractor_calls_a_person_is_rejected():
    """GLiNER labels "she" a person, thirteen mentions and all. A one-word
    candidate the document also writes in lowercase is a common word wearing
    a capital at the start of a sentence."""
    surfaces = {
        record['surface'] for record in BookTranslator.merge_harvested_candidates(ESSAY, [
            {'surface': 'she', 'count': 13, 'kind': 'person', 'evidence': [], 'variants': []},
            {'surface': 'Fanny', 'count': 1, 'kind': 'person', 'evidence': [], 'variants': []},
        ])
    }

    assert 'she' not in surfaces
    assert 'Fanny' in surfaces


def test_rendering_batches_overlap_without_changing_the_result():
    """Stage 0's wall clock is fifteen sequential calls to a large local
    model, and nothing in one batch depends on another's answer. Overlapping
    them is only safe if the merge still sees batch order: the first batch to
    claim a source is the one that keeps it."""
    import threading
    import time

    translator = BookTranslator.__new__(BookTranslator)
    text = 'Grunnings, Privet Drive, Dursley, Dudley, Potters, Surrey, Hogwarts, Fenwick.'
    candidates = [
        {'surface': name, 'count': 3, 'kind': 'other', 'evidence': [], 'variants': []}
        for name in ('Grunnings', 'Privet Drive', 'Dursley', 'Dudley',
                     'Potters', 'Surrey', 'Hogwarts', 'Fenwick')
    ]
    answers = {
        'Grunnings': '[{"source": "Grunnings", "target": "Граннингс"},'
                     ' {"source": "Dursley", "target": "ИЗ ПЕРВОГО БАТЧА"}]',
        'Potters': '[{"source": "Potters", "target": "Поттеры"},'
                   ' {"source": "Dursley", "target": "из второго батча"}]',
    }
    in_flight, peak = set(), []
    lock = threading.Lock()

    def fake_call(prompt, temperature=0.2, read_timeout=0):
        with lock:
            in_flight.add(prompt)
            peak.append(len(in_flight))
        # The batch that comes first answers last, so an out-of-order arrival
        # is what the merge actually has to survive.
        time.sleep(0.05 if 'Grunnings' in prompt else 0.01)
        with lock:
            in_flight.discard(prompt)
        return next((body for key, body in answers.items() if key in prompt), '[]')

    translator._call_model = fake_call
    translator.PREPARE_BATCH_TERMS = 4
    records = translator.propose_proper_noun_records(
        text, 'en', 'ru', 'fiction', candidates=candidates,
    )

    assert max(peak) > 1, 'the batches did not actually overlap'
    by_source = {record['source']: record['target'] for record in records}
    assert by_source['Grunnings'] == 'Граннингс' and by_source['Potters'] == 'Поттеры'
    # Batch order decides the duplicate, not which call happened to finish first.
    assert by_source['Dursley'] == 'ИЗ ПЕРВОГО БАТЧА'


def test_an_answer_without_the_brackets_is_still_an_answer():
    """A model that writes the objects one after another, with no array
    around them, has answered correctly. Reading only the bracketed form
    discarded whole batches of good renderings as "0 of 8 accepted"."""
    unwrapped = """{"source": "Swift", "target": "Свифт", "mode": "inflectable"}
{"source": "Bennet", "target": "Беннет", "mode": "inflectable"}
{"source": "Pride and Prejudice", "target": "Гордость и предубеждение", "mode": "exact"}"""

    items = BookTranslator._parse_json_array(unwrapped)

    assert [item['source'] for item in items] == ['Swift', 'Bennet', 'Pride and Prejudice']


def test_an_array_cut_off_before_its_closing_bracket_keeps_what_arrived():
    items = BookTranslator._parse_json_array(
        '[{"source": "Grunnings", "target": "Граннингс"}, {"source": "Dursley", "targ'
    )

    assert [item['source'] for item in items] == ['Grunnings']


def test_a_proper_array_is_still_read_as_one():
    """The bracketed path is unchanged, nesting and prose included."""
    items = BookTranslator._parse_json_array(
        'Here you go:\n```json\n[{"source": "A", "features": {"nested": 1}}, "junk",'
        ' {"source": "B"}]\n```\nHope that helps.'
    )

    assert [item['source'] for item in items] == ['A', 'B']
    assert items[0]['features'] == {'nested': 1}


def test_an_empty_answer_stays_empty():
    assert BookTranslator._parse_json_array('[]') == []
    assert BookTranslator._parse_json_array('No proper nouns here.') == []
    assert BookTranslator._parse_json_array(None) == []
