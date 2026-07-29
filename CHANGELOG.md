# Changelog

Notable changes to Tolmach are documented here.

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
