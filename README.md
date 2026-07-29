# Tolmach — Offline AI Book Translator for EPUB and Novels

Professional literary translation software for translating entire TXT and EPUB books with local Ollama models, document glossaries, guarded refinement, and side-by-side review.

```bash
git clone https://github.com/KazKozDev/book-translator.git && cd book-translator && python3 launch.py
```

![Tolmach offline AI book translator for EPUB and novels](assets/demo.gif)

Open source · Local by default · No API key required

---

## Quick start

1. Run the command above. It clones the current repository and starts `launch.py`. The launcher creates `venv`, installs the Python dependencies, checks Ollama and the required local models, starts Tolmach at `http://localhost:5001`, and opens it in your browser.

2. Open **Settings**, choose a local model for each role, and click **Save setup**.

3. Return to the main page and follow the numbered buttons:

   ```text
   → 1 UPLOAD → PREPARE (optional) → 2 START → 3 CONTINUE → TXT / PDF / EPUB
   ```

   **START** creates the first translation. **CONTINUE** reviews and refines it. When the job finishes, use the export buttons to download the book.

## Translate an entire EPUB book with AI

Choose the source language, target language, and text genre. Click **→ 1 UPLOAD** and select the complete book—not one chapter at a time.

```text
Source language: English
Target language: Russian
Input:           TXT or EPUB
Download:        TXT, PDF, or EPUB
```

Click **→ 2 START** to create the draft translation. Finished sections appear in the Translation panel while the rest of the book continues processing. Click **→ 3 CONTINUE** when you want Tolmach to refine that draft into the final version.

The job is saved locally, so you can reopen it from the Archive. A complete book can take 10–15 hours; the actual time depends on its length, your models, and your computer.

## Use an AI book translator with a glossary

After uploading the book, click **→ PREPARE**. Tolmach scans the complete source and creates an editable glossary of recurring names, places, organisations, and terms.

```text
Netherfield => Незерфилд | exact
Mr. Darcy => мистер Дарси | inflectable
```

Review this list before starting the translation: delete noise, correct a wrong translation, and add anything the scan missed.

- `exact` keeps the target wording unchanged.
- `inflectable` lets the model change the grammatical form.
- `preferred` tells the model which wording to favor.

The glossary belongs only to this book and language pair. If you configure an optional external provider, **Verify automatically** can check the glossary and show proposed changes; Tolmach applies nothing until you approve it.

## Review a novel translation side by side

After **→ 3 CONTINUE** finishes, open **Review desk**. Each chunk shows the original source, the first draft, and the editable final translation next to each other.

```text
Source → Draft → Final → Save final → Download
```

Start with **Needs review**, inspect the proposed fixes, and apply only the changes you want. You can edit the final text directly or ask for two or three alternatives. Tolmach never replaces your final choice automatically.

Optional quality checks flag possible terminology errors, missing text, changed numbers, wrong-language passages, repetition, and meaning drift. They are diagnostics: they do not rewrite the book or prevent export.

## How it works

The browser sends your TXT or EPUB to a Flask server running on your computer.<br>
**PREPARE** scans the whole book and builds a glossary for that document.<br>
**START** splits the book into chunks and translates them with the selected Ollama model.<br>
**CONTINUE** proposes small edits, and a separate verifier checks each edit against the source.<br>
SQLite saves the job, glossary, aligned text, review state, quality results, and cache locally.

```text
Book → Glossary → Draft translation → Verified refinement → Human review → Download
```

<details>
<summary>Technical architecture</summary>

### Translation pipeline

1. **Upload** — the Flask backend reads TXT or EPUB and stores the source in `uploads/`.
2. **Prepare** — deterministic text harvesting and GLiNER collect entity candidates. BGE-M3 groups likely spelling variants, then the selected instruct model resolves ambiguous groups and proposes target renderings. The editable glossary is stored for this document fingerprint and language pair.
3. **Start** — the source is split into chunks of about 1200 characters at paragraph and sentence boundaries. The Translation model receives each chunk with its genre, glossary constraints, and previous-paragraph context. Completed chunks are written to SQLite and streamed to the browser.
4. **Continue** — the Refinement model returns located edits instead of rewriting an entire chunk. Python applies only those replacements. The Verifier compares each patched version with the source, checks the alternatives in both orders, and retries without ordered A/B versions when it detects position bias.
5. **Review and export** — Review desk keeps Source, Draft, and editable Final text aligned. Exporters write the accepted final text as TXT, PDF, or EPUB.

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

### Storage and quality checks

- `translations.db` stores jobs, aligned chunks, glossary drafts, review state, and saved quality results.
- `cache.db` stores completed chunk translations so resumed jobs do not repeat finished work.
- Deterministic checks inspect the complete document for missing chunks, changed numbers, glossary violations, source-script leakage, unusual length, and repetition.
- Optional model checks include draft/final LLM judging, backtranslation chrF, LaBSE alignment, language identification, and COMET-Kiwi. They report evidence but do not edit the translation.

### Important files

- `launch.py` — cross-platform bootstrap used by the macOS, Linux, and Windows launchers.
- `src/translator.py` — Flask application, API, translation stages, persistence, and export orchestration.
- `src/frontier_glossary.py` — optional external glossary verification.
- `src/quality_tests.py` — deterministic and model-based quality checks.
- `src/terminology.py` — glossary parsing and enforcement rules.
- `src/epub_io.py` — EPUB input and output.
- `src/translation_cache.py` — persistent chunk cache.
- `src/prompts/` — prompts sent to each model role.
- `src/static/` — browser interface, Settings, Guide, and live Log pages.
- `tests/` — model-free unit and integration tests.

</details>

## Configuration

| Setting | Default | What it means |
|---|---|---|
| App address | `http://localhost:5001` | Local browser interface; set `PORT` to change the port |
| Ollama address | `http://localhost:11434` | Local server that runs the language models |
| Translation model | `translategemma:12b` preferred | Creates the first translation during **START** |
| Glossary model | First suitable local instruct model | Builds glossary suggestions during **PREPARE** |
| Refinement model | First suitable local instruct model | Proposes improvements during **CONTINUE** |
| Verifier model | A model different from Refinement | Checks whether proposed edits preserve the source meaning |
| Judge model | A model different from Translation | Runs optional translation-quality diagnostics |
| Chunk size | `1200` characters | Splits first on paragraphs, then on sentence boundaries |
| Source language | English | Chosen separately for each book |
| Target language | Spanish | Chosen separately for each book |
| Text genre | Unknown / Auto | Also supports Fiction, Technical, Academic, Business, and Poetry |
| Glossary verification | Off | Optional: configure `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY` |
| COMET-Kiwi access | Off | Optional: set `HF_TOKEN` after receiving access to the gated model |

## Requirements

- **macOS or Linux** for the one-command installer. Windows users can run the repository launcher.
- **Ollama** running on the same computer.
- The minimum local setup is `translategemma:12b` for translation plus `gemma4:31b` for the other roles. For a more independent refinement check, choose a third instruct model as Verifier.
- Enough memory and disk space for the models you choose.
- Internet access on the first run to download Python dependencies, Ollama models, and optional Hugging Face components.
- Supported languages: English, Russian, Spanish, French, German, Italian, Portuguese, Chinese, Japanese, and Korean.

The installer uses an existing Python 3.10+ installation when available. Otherwise, it installs Python 3.12 through `uv`.

## Limitations

- Tolmach accepts TXT and EPUB input. It does not import PDF or DOCX files.
- Translating a complete book can take 10–15 hours on local hardware.
- Translation and review models can still miss errors or make good text worse. Proofread the final book before publishing it.
- Large Ollama models need substantial memory and disk space; Tolmach cannot make a model fit hardware that is too small.
- Optional glossary verification sends the glossary and language pair—not the full book—to the selected API provider and may cost money.
- COMET-Kiwi is optional, downloads a multi-gigabyte gated checkpoint, and requires Hugging Face access.
- Tolmach does not currently provide an official Docker image.

<details>
<summary>Manual installation, Docker, development setup</summary>

### Manual installation

```bash
git clone https://github.com/KazKozDev/book-translator.git
cd book-translator
python3 launch.py
```

`launch.py` creates the virtual environment, installs runtime dependencies, checks Ollama and the required models, starts the server, and opens the browser.

The platform launchers use the same setup:

- macOS: double-click `Launch Book-Translator.command`
- Linux: run `./Launch Book-Translator.sh`
- Windows: double-click `Launch Book-Translator.bat`

To use optional glossary verification, copy `.env.example` to `.env.local` and add the provider key. On macOS and Linux, set the file permissions to `600`.

### Docker

This repository does not currently include a Dockerfile or published image. Use the native launcher so Tolmach can detect Ollama, installed models, and local hardware.

### Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
ruff check .
```

The test suite does not require Ollama, downloaded models, or network access.

</details>

## Credits and project history

[**kroryan**](https://github.com/kroryan) — created the v2.0 modular architecture, contributed Windows compatibility, and identified security and stability fixes later carried into this rewrite.

[**moonixt**](https://github.com/moonixt) — contributed Portuguese language support and startup instructions.

The earlier modular application remains available as tag **v2.1.0**, including its Windows tray application and Docker build as they stood on 26-01-21.

<br>

<p align="center">
  <a href="https://github.com/KazKozDev/book-translator/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
  <a href="https://github.com/KazKozDev/book-translator/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/KazKozDev/book-translator/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://www.python.org/"><img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&amp;logoColor=white"></a>
</p>

<p align="center">
  <a href="https://github.com/KazKozDev/book-translator/issues">Issues</a> ·
  <a href="https://github.com/KazKozDev/book-translator/blob/main/CHANGELOG.md">CHANGELOG</a> ·
  <a href="https://github.com/KazKozDev/book-translator/blob/main/CONTRIBUTING.md">CONTRIBUTING</a> ·
  <a href="https://github.com/KazKozDev/book-translator/blob/main/LICENSE">LICENSE</a> ·
  <a href="https://github.com/KazKozDev/book-translator/blob/main/SECURITY.md"><strong>Disclaimer</strong></a> ·
  <a href="https://www.linkedin.com/in/kazkozdev/">LinkedIn</a>
</p>
