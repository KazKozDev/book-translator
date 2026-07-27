# Contributing

Thanks for contributing to Book Translator.

## What this project is, so a PR is not wasted

The direction is set by one person, and it has changed by rewrites rather than
by growth: the modular v2 line was replaced wholesale by this pipeline, and its
contributors' code did not survive that. Their findings did — see the Credits in
the [README](README.md), and the [`v2.1.0`](https://github.com/KazKozDev/book-translator/releases/tag/v2.1.0)
release, which keeps that line installable.

So before writing anything large, open an issue and ask. A focused fix, a
language addition, or a bug with a reproduction is always welcome. An
architecture change is worth a conversation first.

## Before you start

- Read the [README](README.md) and get the project running locally.
- Python 3.10 or newer. The launcher refuses to start below that, and `gliner` has no wheels for 3.9.
- [Ollama](https://ollama.com/) installed and running, if your change touches
  anything that calls a model.

## Local setup

```bash
git clone https://github.com/KazKozDev/book-translator.git
cd book-translator
pip install -r requirements-dev.txt
python -m pytest tests -q
python launch.py
```

`requirements.txt` is what the app needs; `requirements-dev.txt` adds the test
and lint tooling and pulls the former in. `python launch.py` does the whole
setup — virtual environment, dependencies, Ollama check — and is the same thing
the `Launch Book-Translator.*` files run.

## Branches and commits

- Branch from `main`.
- Keep changes focused. Small PRs are easier to review and safer to merge.
- Say *why* in the commit message, not only *what*. The reason a change exists
  is the part that cannot be recovered from the diff later.

## What a change has to come with

- `python -m pytest tests -q` passing. The suite is deliberately model-free:
  everything it covers is decidable without Ollama, and that is what keeps it
  fast enough to run on every commit.
- A test for anything whose failure would be silent. This project is full of
  such places — a wrong encoding guess returns plausible nonsense rather than an
  error, a refinement pass that rejects every patch looks exactly like a clean
  draft. If your change has a failure mode that produces no error message, pin
  it with a test.
- Documentation updated when behaviour changes, in the README and in
  `static/guide.html`, which is the in-app guide.

## Reporting bugs

Use the issue templates. For anything involving a translation run, the log is
the useful attachment: open the **Log** window from the app's header, or copy
`logs/translations.log`. It records what each pass found, patched, and decided,
and by which model.

Security issues go to `kazkozdev@gmail.com` privately — see
[SECURITY.md](SECURITY.md).
