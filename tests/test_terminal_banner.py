import translator


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
