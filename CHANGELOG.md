# Changelog

Notable changes to Tolmach are documented here.

## Unreleased

- Added a **Refinement mode** setting for **CONTINUE**: apply verified fixes automatically (the default), *Suggest only* — every proposed fix waits in the Review desk and nothing is written into the text without you, or *Skip refinement* — the draft becomes the final text without a model call (#27).
- Review desk has a **Restore draft** button that puts a chunk's draft back into Final.
- Added **shared glossaries**: name one above the glossary and the entries you approve carry over to the next document in the same language pair — the next chapter of a novel, say. **PREPARE** fills in the saved entry for every name it finds in the new text and lists those last; names that do not occur in the text are not added. **START** saves the approved list back under that name, and any entry can be ticked to keep it out; the ticks are remembered with the document (#28).

## [3.1.0] — 2026-10-02

- Changed the license from MIT to AGPL-3.0-only.
- Added DOCX upload. Heading 1 sections become chapters when the document has them.
- PDF uploads keep chapter breaks when the file has bookmarks or detectable headings.
- A long Start run keeps going when the browser tab is closed, and **Resume** in History continues from the last finished chunk after an interrupt or a server restart.
- Review desk can hand proposed fixes to the configured cloud provider: **Review chunk automatically** decides apply or keep for one chunk and saves it, and **Review all automatically** does the same for every open chunk. The provider can only accept or decline the fixes already proposed; it cannot rewrite a chunk.
- Fixed CONTINUE staying disabled after the draft translation finished.
- The failed-translations list loads while Ollama is stopped.
- Fixed paragraphs left in the source language. The review pass has its own `untranslated` category — reported and patched at any severity, because whether a passage is translated is not a matter of degree — and a chunk that is still written in the source language is now named to the reviewer outright, since an untranslated paragraph does not look wrong when it is printed under its own source. Guards refuse a replacement that reproduces the source text, paraphrases it back into the source language, puts a source word back as a glossary rendering, or collapses a long span to a fraction of its length; the length guard now compares across scripts, so a correct Chinese rendering of a paragraph is not mistaken for a deletion.
- Fixed paragraphs printed twice in the refined text. A reported error whose replacement restates the passage that follows its span — a five-sentence paragraph, then the same five sentences — is refused, as is an `omission`/`addition`/`terminology`/`consistency` patch that together covers more than a quarter of the draft, which is a rewrite wearing a category's clothes and now goes to the verifier like any other. The guard that refuses a source word put back as a rendering now stands down for a name the glossary agreed to, so `Rom => Rom` and an English name kept in a French page are no longer dropped as the very mistake it prevents. Refused edits are counted per reason and reported in the chunk log and the Refinement panel. A span the reviewer reported that is not in the draft at all is counted too, under `span not in draft`, so a chunk whose whole review answer missed the text no longer reads as "0 found".
- The language guard is tuned against its measurements rather than one book: the bar sits above the worst false positive observed on the French corpus, words shared between a source language and the language it is translated into are no longer markers, and the Spanish, Italian, Portuguese and Russian marker lists were brought up to a size where a genuine untranslated passage is actually caught in them. Chinese and Japanese have no entry at all, because a word list cannot be read out of text with no spaces. A Chinese or Japanese source is still covered once the reviewer has proposed a replacement — the run check compares that replacement against the source and needs no word boundaries — but the pass cannot name such a chunk as untranslated by itself, since detecting that is exactly what the word list is for. A run where source and target are the same language stands every one of these guards down.
- **Copy frontier prompt** now names the book from the file's own title and author metadata and adds a second task asking the model to research the lore and propose a `{note}` where it is certain. A TXT or DOCX upload carries no title or author, so it says the book is unknown and lets the model identify the work from the entries rather than guess from the filename. **Verify automatically** is unchanged and still never sees a note.
- Added an optional per-term glossary note: `Rom => Rom | inflectable {c'est un garçon}`. A note is free text for what the author knows that the book does not state outright — the gender of a name, an ambiguity in the original — and it reaches the Translation and Refinement prompts. Notes are never sent to an external verifier and can never be changed or invented by one. Only `exact` remains deterministically checkable; a note is guidance, not a rule. Editing a note retires the cached chunks, since it changes the prompt.
- Added PDF upload beside TXT and EPUB. A PDF is read as text only: running heads and page numbers are removed and printed lines are rejoined into paragraphs, after which it follows the same path as a TXT book. A scanned PDF with no text layer is refused instead of translated as an empty book.
- **Prepare reports progress and any running job can be paused.** The glossary scan used to hold the browser on a blank screen until it finished. It now streams its four stages into the job rail — extracting names, resolving which source forms name one entity, proposing target renderings, checking for rendering conflicts — weighted so the bar advances smoothly instead of jumping, and it reports each batch of the scan as it goes. A Pause button beside the bar suspends Prepare, Start, or Continue at its next chunk or batch boundary, which releases the GPU without persisting anything: no state is written while a job is paused, so stopping the process then simply loses the chunk that had not been saved yet. Resuming continues from that boundary. The button works with Ollama stopped, because it controls a worker that is already running and not a request that needs the service.
- `/prepare` answers with a Server-Sent Events stream instead of a single JSON object, so that it can report progress. This changes the route's contract: the glossary arrives in the event whose `stage` is `completed`, and a failure arrives as an event carrying `error` on an otherwise successful response, where a Stage 0 failure used to be an HTTP 503 with the message in the body. `index.html` was updated to match.

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

[3.1.0]: https://github.com/KazKozDev/book-translator/releases/tag/v3.1.0
[3.0.1]: https://github.com/KazKozDev/book-translator/releases/tag/v3.0.1
[3.0.0]: https://github.com/KazKozDev/book-translator/releases/tag/v3.0.0
[2.1.0]: https://github.com/KazKozDev/book-translator/releases/tag/v2.1.0
[2.0.0]: https://github.com/KazKozDev/book-translator/releases/tag/v2.0.0
