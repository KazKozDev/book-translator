from translator import BookTranslator, TerminologyManager, is_translategemma


# The prompt exactly as TranslateGemma's model card documents it, including the
# two blank lines before the text. Written out in full so a change to the
# builder that drifts from the card fails here instead of silently degrading
# every translation.
CANONICAL_EN_RU = (
    "You are a professional English (en) to Russian (ru) translator. Your goal "
    "is to accurately convey the meaning and nuances of the original English "
    "text while adhering to Russian grammar, vocabulary, and cultural "
    "sensitivities.\n"
    "Produce only the Russian translation, without any additional explanations "
    "or commentary. Please translate the following English text into Russian:\n"
    "\n"
    "\n"
    "The garden was quiet."
)


def build(**kwargs):
    params = {
        "text": "The garden was quiet.",
        "source_lang": "en",
        "target_lang": "ru",
        "source_name": "English",
        "target_name": "Russian",
        "previous_chunk": "",
        "genre": "unknown",
        "terminology_context": "",
    }
    params.update(kwargs)
    return BookTranslator._stage1_prompt_translategemma(**params)


def test_detects_translategemma_by_name():
    assert is_translategemma("translategemma:27b")
    assert is_translategemma("TranslateGemma:4b")
    assert not is_translategemma("qwen3:4b-instruct")
    assert not is_translategemma("gemma3:12b")
    assert not is_translategemma("")
    assert not is_translategemma(None)


def test_prompt_without_extras_matches_the_model_card_byte_for_byte():
    assert build() == CANONICAL_EN_RU


def test_extras_sit_between_the_framing_sentences_and_text_stays_last():
    terminology = TerminologyManager.from_text("garden => сад | exact")
    prompt = build(
        previous_chunk="Дом стоял на холме.",
        genre="fiction",
        terminology_context=terminology.prompt_context("The garden was quiet."),
    )

    opening, rest = prompt.split("\n", 1)
    assert opening.startswith("You are a professional English (en) to Russian (ru) translator.")
    assert rest.startswith(
        "Produce only the Russian translation, without any additional "
        "explanations or commentary.\n\n"
    )

    # Everything extra is above the instruction sentence...
    middle, tail = rest.split(
        "\nPlease translate the following English text into Russian:", 1
    )
    assert "- Document type: fiction" in middle
    assert "Дом стоял на холме." in middle
    assert '- "garden" => "сад" (use this target form exactly)' in middle

    # ...and the tail the model was trained on is untouched: two blank lines,
    # the chunk, nothing after it.
    assert tail == "\n\n\nThe garden was quiet."


def test_genre_line_is_omitted_when_unknown():
    prompt = build(previous_chunk="Дом стоял на холме.", genre="unknown")

    assert "Document type" not in prompt
    assert "- Preserve formatting (paragraphs, line breaks)" in prompt


def test_payload_carries_translategemma_sampling_options_only_for_that_model():
    options = BookTranslator("translategemma:27b")._ollama_payload("hi", 0.3)["options"]
    assert options == {
        "temperature": 0.3,
        "num_ctx": 8192,
        "top_k": 64,
        "top_p": 0.95,
        "stop": ["<end_of_turn>"],
    }

    other = BookTranslator("qwen3:4b-instruct")._ollama_payload("hi", 0.6)["options"]
    assert other == {"temperature": 0.6, "num_ctx": 8192}


def test_chunks_stay_within_the_configured_size():
    paragraph = "Alice walked through the quiet garden. " * 60  # ~2280 chars
    text = "\n\n".join([paragraph] * 4)

    chunks = BookTranslator("qwen3:4b-instruct").split_into_chunks(text)

    assert chunks
    assert max(len(chunk) for chunk in chunks) <= 1200
