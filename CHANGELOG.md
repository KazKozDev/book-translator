# Changelog

Notable changes to Tolmach are documented here.

## Unreleased

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
