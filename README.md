<h1 align="center">Tolmach</h1>

<p align="center">
  <b>Translate a whole novel on your own computer.</b><br>
  Consistent names. Verified edits. Your text never leaves your machine.
</p>

<p align="center">
  EPUB · PDF · DOCX · TXT &nbsp;→&nbsp; a finished book in your language, ready to read.
</p>

<p align="center">
  <a href="https://github.com/KazKozDev/book-translator/archive/refs/heads/main.zip"><b>Download</b></a> ·
  <a href="#get-started-in-one-command">Get started</a> ·
  <a href="https://www.youtube.com/watch?v=-lMNAKOp1Kc">Watch the demo</a>
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/eae8eae5-6ca9-4b2c-8f67-cf7a39848794" controls muted playsinline width="820">
    Your browser does not support inline video.
    <a href="https://www.youtube.com/watch?v=-lMNAKOp1Kc" target="_blank" rel="noopener noreferrer">Watch on YouTube</a>
  </video>
</p>

<p align="center">
  <img alt="Local-first" src="https://img.shields.io/badge/Local--first-Ollama-black">
  <img alt="Offline" src="https://img.shields.io/badge/Private-no%20cloud%20required-2ea44f">
  <img alt="Languages" src="https://img.shields.io/badge/Languages-11-blue">
  <a href="LICENSE"><img alt="License: AGPL-3.0-only" src="https://img.shields.io/badge/License-AGPL--3.0--only-blue.svg"></a>
  <a href="https://github.com/KazKozDev/book-translator/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/KazKozDev/book-translator/actions/workflows/ci.yml/badge.svg"></a>
</p>

---

## Get started in one command

```bash
# macOS / Linux
git clone https://github.com/KazKozDev/book-translator.git && cd book-translator && python3 launch.py
```

```bash
# Windows
git clone https://github.com/KazKozDev/book-translator.git
cd book-translator
py -3 launch.py
```

<p align="center">
  <a href="Launch%20Book-Translator.command"><img src="assets/badges/macos.png" alt="macOS" height="36"></a>
  <a href="Launch%20Book-Translator.bat"><img src="assets/badges/windows.png" alt="Windows" height="36"></a>
  <a href="Launch%20Book-Translator.sh"><img src="assets/badges/linux.png" alt="Linux" height="36"></a>
</p>

The launcher creates a virtual environment, installs dependencies, checks Ollama and models, starts Tolmach at `http://localhost:5001`, and opens your browser.

Then: **Settings** → pick a model per role → **Save setup** → **1 UPLOAD** → **2 START** → **3 CONTINUE** → export.

---

## Chat translators break books

Paste a chapter into a chatbot and you get a decent page. Do it for 400 pages and the problems show up:

| Without Tolmach | With Tolmach |
|---|---|
| The hero's name changes between chapters | A glossary built from the whole book locks every name, place, and term |
| "Improve this" passes rewrite good wording | Only small, located edits — each checked against the source |
| Untranslated passages slip through | Source-language leftovers are detected and patched |
| You paste chunks, then reassemble by hand | Upload the book, download the book |
| Your manuscript goes to someone else's server | Everything runs locally through Ollama |

---

## How it works

```text
Upload → Prepare → Start → Continue → Review → Download
```

<table>
  <tr>
    <td width="33%" valign="top">
      <h3>1. Prepare</h3>
      Scans the entire book and proposes a glossary of characters, places, organisations, and terms. You approve it before anything is translated.
    </td>
    <td width="33%" valign="top">
      <h3>2. Start</h3>
      Translates in chunks with genre, glossary rules, and the previous paragraph as context. Finished sections appear while the rest keeps running.
    </td>
    <td width="33%" valign="top">
      <h3>3. Continue</h3>
      A refinement model proposes small edits. A separate verifier checks each one against the source before it touches the text.
    </td>
  </tr>
</table>

Then open the **Review desk**: source, first draft, and editable final text side by side. Export to **TXT, PDF, or EPUB**.

---

## Features

**Glossary that stays consistent**
- Built from the full text, not one chapter at a time
- `exact`, `inflectable`, and `preferred` rules per term
- Notes like `{he is a boy}` that the translator obeys
- Reuse one glossary across chapters of the same novel: name it under **Shared glossary**, and **PREPARE** fills in entries you approved before for names in the new text
- **Copy frontier prompt** asks an external model to research the lore and write notes, only where it is certain. Read them before pasting back: a wrong note is one the translator will obey

```text
Netherfield => Незерфилд | exact
Mr. Darcy   => мистер Дарси | inflectable
```

**Edits that cannot ruin good text**
- The model returns located fixes, not a rewritten chunk
- Python rejects edits that leak source text, duplicate a passage, or delete most of a span
- A verifier checks both orderings to avoid position bias

**You stay in control**
- Review desk with *Needs review* filter
- Apply fixes one by one, or let a cloud provider decide apply/keep
- Ask for 2–3 alternatives on any passage
- Refinement mode: *auto-apply*, *suggest only*, or *skip*
- **Pause** any job (Prepare, Start, or Continue) at the next chunk boundary to free the GPU, then press it again to carry on
- **Resume** from History after an interrupt or server restart; the job continues from the last finished chunk
- Everything is saved locally, so any job can be reopened from the Archive

**Quality diagnostics**
- Missing text, changed numbers, glossary violations, repetition, wrong-language passages
- Optional LLM judge, backtranslation chrF, LaBSE alignment, COMET-Kiwi
- Reports only — never rewrites your book

**Books in, books out**
- Input: TXT, EPUB, PDF, DOCX. Output: TXT, PDF, EPUB
- PDF is read as text: running heads and page numbers are removed, printed lines are rejoined into paragraphs, and bookmarks or clear headings become chapters
- DOCX keeps Heading 1 sections as chapters
- A scanned PDF with no text layer is refused: run OCR first, or use TXT, EPUB, or DOCX

**Private by design**
- Runs on `localhost` with Ollama and SQLite
- Cloud glossary verification is optional and sends only the glossary and language pair, never the book

---

## 11 languages, one desk

Any of 11 languages as source or target: English, Russian, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, Korean, and Turkish. Screenshots below show English → nine of them; click a thumbnail for the full size.

<table>
  <tr>
    <td align="center"><a href="assets/locales/ru_RU.png"><img src="assets/locales/thumbs/ru_RU.png" alt="English to Russian" width="260"></a><br><code>ru_RU</code></td>
    <td align="center"><a href="assets/locales/es_ES.png"><img src="assets/locales/thumbs/es_ES.png" alt="English to Spanish" width="260"></a><br><code>es_ES</code></td>
    <td align="center"><a href="assets/locales/fr_FR.png"><img src="assets/locales/thumbs/fr_FR.png" alt="English to French" width="260"></a><br><code>fr_FR</code></td>
  </tr>
  <tr>
    <td align="center"><a href="assets/locales/de_DE.png"><img src="assets/locales/thumbs/de_DE.png" alt="English to German" width="260"></a><br><code>de_DE</code></td>
    <td align="center"><a href="assets/locales/it_IT.png"><img src="assets/locales/thumbs/it_IT.png" alt="English to Italian" width="260"></a><br><code>it_IT</code></td>
    <td align="center"><a href="assets/locales/pt_BR.png"><img src="assets/locales/thumbs/pt_BR.png" alt="English to Portuguese" width="260"></a><br><code>pt_BR</code></td>
  </tr>
  <tr>
    <td align="center"><a href="assets/locales/zh_CN.png"><img src="assets/locales/thumbs/zh_CN.png" alt="English to Chinese" width="260"></a><br><code>zh_CN</code></td>
    <td align="center"><a href="assets/locales/ja_JP.png"><img src="assets/locales/thumbs/ja_JP.png" alt="English to Japanese" width="260"></a><br><code>ja_JP</code></td>
    <td align="center"><a href="assets/locales/ko_KR.png"><img src="assets/locales/thumbs/ko_KR.png" alt="English to Korean" width="260"></a><br><code>ko_KR</code></td>
  </tr>
</table>

---

## What you need

- **Ollama** on the same computer
- Minimum models: `translategemma:12b` for translation plus `gemma4:31b` for the other roles
- Python 3.10+ (installed automatically via `uv` if missing)
- Memory and disk space for the models you choose

**Recommended stack** (author's tested setup, cloud open-source models):

| Role | Model |
|---|---|
| Glossary, Refinement | `gemma4:31b-cloud` |
| Translation | `translategemma:27b` |
| Verifier, Judge | `mistral-large-4:cloud` |

---

## Honest expectations

- A full book takes **10–15 hours** on local hardware. This is a pipeline, not a chat reply.
- Quality depends on your models. Smaller ones run, with lower quality.
- PDFs are read as text only; scanned PDFs without a text layer are refused. Layout and images are not preserved.
- Models can still miss errors. Proofread before you publish.
- No official Docker image.

---

<details>
<summary><b>Under the hood</b></summary>

```text
Browser
   ↓
Flask UI + JSON API
   ↓
Prepare → Start → Continue → Review
   ↓         ↓         ↓
Glossary   Ollama   Verifier
   └───────── SQLite ─────────┘
                ↓
          TXT / PDF / EPUB
```

- **Prepare** — text harvesting and GLiNER collect candidates; BGE-M3 groups spelling variants; an instruct model proposes renderings.
- **Start** — chunks of about 1200 characters split at paragraph and sentence boundaries, streamed to SQLite and the browser.
- **Continue** — located edits, deterministic guards, verifier with position-bias retry.
- **Storage** — `translations.db` (jobs, chunks, glossary, review state), `cache.db` (chunk cache for resumes).

Key files: `launch.py`, `src/translator.py`, `src/terminology.py`, `src/quality_tests.py`, `src/epub_io.py`, `src/pdf_io.py`, `src/frontier_glossary.py`, `src/translation_cache.py`, `src/prompts/`, `tests/`.

</details>

<details>
<summary><b>Configuration</b></summary>

| Setting | Default | Meaning |
|---|---|---|
| App address | `http://localhost:5001` | Set `PORT` to change |
| Reachable from | This computer only | `HOST=0.0.0.0` opens it to your network (no login — trusted networks only) |
| Ollama address | `http://localhost:11434` | Local model server |
| Translation model | `translategemma:12b` preferred | Used by **START** |
| Glossary / Refinement | First suitable instruct model | Used by **PREPARE** / **CONTINUE** |
| Verifier | Different from Refinement | Checks proposed edits |
| Judge | Different from Translation | Optional diagnostics |
| Chunk size | `1200` characters | Paragraphs first, then sentences |
| Genre | Auto | Fiction, Technical, Academic, Business, Poetry |
| Glossary verification | Off | Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY` |
| COMET-Kiwi | Off | Set `HF_TOKEN` after gated-model access |

</details>

<details>
<summary><b>Development</b></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
ruff check .
```

Tests need no Ollama, models, or network.

</details>

---

## Contributors

- [@StellarNear](https://github.com/StellarNear) — glossary notes, pause/resume, Prepare progress, Stage 2 guards ([#23](https://github.com/KazKozDev/book-translator/pull/23))
- [@kroryan](https://github.com/kroryan) — Windows build, Korean support, v2 refactoring ([#9](https://github.com/KazKozDev/book-translator/pull/9))
- [@moonixt](https://github.com/moonixt) — Portuguese support ([#6](https://github.com/KazKozDev/book-translator/pull/6))

## License

[AGPL-3.0-only](LICENSE)

<p align="center">
  <a href="https://github.com/KazKozDev/book-translator/issues">Issues</a> ·
  <a href="CHANGELOG.md">Changelog</a> ·
  <a href="CONTRIBUTING.md">Contributing</a> ·
  <a href="DISCLAIMER.md">Disclaimer</a> ·
  <a href="https://www.linkedin.com/in/kazkozdev/">LinkedIn</a>
</p>
