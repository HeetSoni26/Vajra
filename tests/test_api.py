"""Tests for the FastAPI server endpoints."""

from __future__ import annotations

import os
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def api_client():
    """Create a TestClient with lifespan that initialises the engine."""
    os.environ["FLM_CONFIG"] = "configs/training/pretrain_tiny.yaml"
    os.environ["FLM_CHECKPOINT"] = ""
    from api.main import app
    with TestClient(app) as client:
        yield client


def test_health_endpoint(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_models_endpoint(api_client):
    resp = api_client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data
    assert len(data["data"]) > 0


def test_model_info_endpoint(api_client):
    resp = api_client.get("/model")
    assert resp.status_code == 200
    info = resp.json()
    assert "total_parameters" in info


def test_generate_endpoint(api_client):
    resp = api_client.post("/generate", json={
        "prompt": "Hello",
        "max_tokens": 5,
        "temperature": 0.0,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "choices" in data
    assert len(data["choices"]) > 0


def test_tokenize_endpoint(api_client):
    resp = api_client.post("/tokenize", json={"text": "Hello world"})
    assert resp.status_code == 200
    data = resp.json()
    assert "ids" in data
    assert "num_tokens" in data


def test_detokenize_endpoint(api_client):
    # First tokenize to get valid IDs
    tok_resp = api_client.post("/tokenize", json={"text": "Hello"})
    ids = tok_resp.json()["ids"]
    resp = api_client.post("/detokenize", json={"ids": ids})
    assert resp.status_code == 200
    assert "text" in resp.json()
