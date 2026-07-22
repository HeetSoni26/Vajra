import pytest

fastapi = pytest.importorskip("fastapi")


def test_api_imports():
    from api.main import app

    assert app.title == "Foundation LM API"
