from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_starts_without_api_key():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=30).run()

    assert not app.exception
    assert len(app.title) == 1
    assert len(app.text_input) == 3
    assert len(app.chat_input) == 1
    assert app.chat_input[0].disabled is True
