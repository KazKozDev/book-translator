import banner
import translator


def test_every_row_of_the_logo_is_the_same_width():
    """A row one cell short is invisible in a diff and obvious on screen: the
    corner of the last letter sits under the wrong column."""
    rows = banner.TERMINAL_LOGO.split('\n')

    assert len(rows) == 6
    assert {len(row) for row in rows} == {61}
    # Every row must close its own boxes: no row may end mid-glyph.
    assert all(row[-1] in '╗║╝' for row in rows)


def test_the_banner_has_one_source():
    """The launcher cannot import translator (it runs before the venv exists),
    so the artwork lives in its own module and both entry points use it."""
    assert translator.TERMINAL_LOGO is banner.TERMINAL_LOGO
    assert translator.print_terminal_banner is banner.print_terminal_banner


def test_direct_python_banner_matches_tolmach_identity(monkeypatch, capsys):
    monkeypatch.delenv('TOLMACH_BANNER_PRINTED', raising=False)
    monkeypatch.setattr(translator.sys.stdout, 'isatty', lambda: False)

    translator.print_terminal_banner()

    output = capsys.readouterr().out
    assert output.startswith('\n\n\n████████╗')
    assert 'B O O K   T R A N S L A T O R  v3.0' in output
    assert output.endswith('\n\n\n')


def test_launcher_marker_prevents_a_duplicate_banner(monkeypatch, capsys):
    monkeypatch.setenv('TOLMACH_BANNER_PRINTED', '1')

    translator.print_terminal_banner()

    assert capsys.readouterr().out == ''


def test_command_launcher_keeps_the_banner_colored_even_with_global_no_color(
    monkeypatch, capsys,
):
    """Finder/Codex may pass NO_COLOR to Terminal; the .command logo is visual."""
    monkeypatch.delenv('TOLMACH_BANNER_PRINTED', raising=False)
    monkeypatch.setenv('NO_COLOR', '1')
    monkeypatch.setenv('TOLMACH_FORCE_BANNER_COLOR', '1')
    monkeypatch.setattr(banner.sys.stdout, 'isatty', lambda: True)

    banner.print_terminal_banner(force=True)

    output = capsys.readouterr().out
    assert f'{banner._ACCENT}{banner.TERMINAL_LOGO}{banner._RESET}' in output
    assert f'{banner._CREAM}' in output
