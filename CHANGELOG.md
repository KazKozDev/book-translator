# Changelog

Notable changes to Tolmach are documented here.

## Unreleased

- Fixed paragraphs left in the source language. The review pass has its own `untranslated` category — reported and patched at any severity, because whether a passage is translated is not a matter of degree — and a chunk that is still written in the source language is now named to the reviewer outright, since an untranslated paragraph does not look wrong when it is printed under its own source. Guards refuse a replacement that reproduces the source text, paraphrases it back into the source language, puts a source word back as a glossary rendering, or collapses a long span to a fraction of its length; the length guard now compares across scripts, so a correct Chinese rendering of a paragraph is not mistaken for a deletion.
- Fixed paragraphs printed twice in the refined text. A reported error whose replacement restates the passage that follows its span — a five-sentence paragraph, then the same five sentences — is refused, as is an `omission`/`addition`/`terminology`/`consistency` patch that together covers more than a quarter of the draft, which is a rewrite wearing a category's clothes and now goes to the verifier like any other. Refused edits are counted per guard and reported in the chunk log, so a chunk whose whole review answer was dropped no longer reads as "0 found".
- The language guard is tuned against its measurements rather than one book: the bar now sits above the worst false positive observed on the French corpus, words shared between a source language and the language it is translated into are no longer markers, and there is no Chinese or Japanese entry at all — a word list cannot be read out of text with no spaces, and Chinese and Japanese source text is still covered by the run check, which needs no word boundaries. A run where source and target are the same language stands every one of these guards down.
- **Copy frontier prompt** now names the book from the file's own title and author metadata and adds a second task asking the model to research the lore and propose a `{note}` where it is certain. A TXT upload has no metadata, so it says the book is unknown and lets the model identify the work from the entries rather than guess from the filename. **Verify automatically** is unchanged and still never sees a note.
- Added an optional per-term glossary note: `Rom => Rom | inflectable {c'est un garçon}`. A note is free text for what the author knows that the book does not state outright — the gender of a name, an ambiguity in the original — and it reaches the Translation and Refinement prompts. Notes are never sent to an external verifier and can never be changed or invented by one. Only `exact` remains deterministically checkable; a note is guidance, not a rule. Editing a note retires the cached chunks, since it changes the prompt.
- Added PDF upload beside TXT and EPUB. A PDF is read as text only: running heads and page numbers are removed and printed lines are rejoined into paragraphs, after which it follows the same path as a TXT book. A scanned PDF with no text layer is refused instead of translated as an empty book.

## [3.0.1] — 2026-07-29

- Kept source preview and document glossary storage available when Ollama is stopped.
- Made the test suite portable across Linux, macOS, and Windows.
- Published the verified 3.0 release after the complete CI matrix passed.

## [3.0.0] — 2026-07-29

- Rebuilt the application around the **Prepare → Start → Continue** workflow.
- Added document-specific glossary preparation and optional external verification.
- Added guarded refinement with located patches and a separate verifier.
- Added Review desk with aligned Source, Draft, and Final text.
- Added document-level and model-based quality checks.
- Added persistent jobs, glossary drafts, review state, and translation cache.
- Added TXT and EPUB input with TXT, PDF, and EPUB export.
- Added the cross-platform `launch.py` bootstrap, macOS/Linux installer, new README, and GIF demo.

## [2.1.0] — 2026-01-21

- Preserved the modular v2 application, Windows tray app, Docker build, CI workflow, and contributed security and stability fixes.

## [2.0.0] — 2025-10-04

- Completed the earlier project rewrite and modular architecture.

[3.0.1]: https://github.com/KazKozDev/book-translator/releases/tag/v3.0.1
[3.0.0]: https://github.com/KazKozDev/book-translator/releases/tag/v3.0.0
[2.1.0]: https://github.com/KazKozDev/book-translator/releases/tag/v2.1.0
[2.0.0]: https://github.com/KazKozDev/book-translator/releases/tag/v2.0.0
