"""UI contract tests for M6 (SPEC M6): degradation without the API, and the
no-internals rule. Screen behaviour with a live API is exercised manually and
in the Phase 4 clean-machine check."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app" / "main.py"


def test_stopped_api_shows_banner_not_traceback(monkeypatch) -> None:
    # Nothing listens on this port; every screen must degrade to the banner.
    monkeypatch.setenv("RWSAT_API_URL", "http://127.0.0.1:59999")
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    assert not at.exception, f"UI raised instead of degrading: {at.exception}"
    banners = [e.value for e in at.error]
    assert any("docker compose up" in text for text in banners)
    assert any("uv run uvicorn rwsat.api:app" in text for text in banners)


def test_ui_imports_no_rwsat_internals() -> None:
    import ast

    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    offenders = [name for name in imported if name == "rwsat" or name.startswith("rwsat.")]
    assert offenders == [], f"UI must talk HTTP only, but imports {offenders}"
