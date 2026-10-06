import importlib

readme = importlib.import_module("experiments.06_readme")


def test_readme_tables_match_results():
    """Every table in README.md is rendered from results/; none is typed by hand."""
    text = readme.README.read_text()
    assert readme.render(text) == text, "run `make readme`"
