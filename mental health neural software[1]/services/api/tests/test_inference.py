"""
Pytest tests for MindWatch API
"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "MindWatch API"


def test_health():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_inference_with_text():
    """Test inference with text embedding"""
    request = {
        "ht": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 2,  # 16 dims
        "meta": {
            "sessionId": "test-session",
            "consent": {
                "vision": False,
                "audio": False,
                "text": True,
                "context": False,
                "logs": False,
            },
            "timestamp": "2024-01-01T00:00:00Z",
        },
    }
    response = client.post("/infer", json=request)
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "label" in data
    assert data["label"] in ["GREEN", "AMBER", "RED"]
    assert 0 <= data["score"] <= 1
    assert "explanation" in data
    assert "modalityWeights" in data
    assert "topFactors" in data


def test_inference_with_all_modalities():
    """Test inference with all modalities"""
    request = {
        "hv": [0.1] * 32,  # Vision embedding
        "ha": [0.2] * 32,  # Audio embedding
        "ht": [0.3] * 16,  # Text embedding
        "hc": [0.4] * 8,  # Context embedding
        "meta": {
            "sessionId": "test-session",
            "consent": {
                "vision": True,
                "audio": True,
                "text": True,
                "context": True,
                "logs": False,
            },
            "timestamp": "2024-01-01T00:00:00Z",
        },
    }
    response = client.post("/infer", json=request)
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "label" in data
    assert data["label"] in ["GREEN", "AMBER", "RED"]


def test_inference_no_modalities():
    """Test inference with no modalities (should fail)"""
    request = {
        "meta": {
            "sessionId": "test-session",
            "consent": {
                "vision": False,
                "audio": False,
                "text": False,
                "context": False,
                "logs": False,
            },
            "timestamp": "2024-01-01T00:00:00Z",
        },
    }
    response = client.post("/infer", json=request)
    assert response.status_code == 400
    assert "at least one modality" in response.json()["detail"].lower()


def test_inference_with_vision():
    """Test inference with vision embedding"""
    request = {
        "hv": [0.1, 0.2, 0.3, 0.4, 0.5] * 7,  # 32 dims (approx)
        "meta": {
            "sessionId": "test-session",
            "consent": {
                "vision": True,
                "audio": False,
                "text": False,
                "context": False,
                "logs": False,
            },
            "timestamp": "2024-01-01T00:00:00Z",
        },
    }
    response = client.post("/infer", json=request)
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "modalityWeights" in data
    assert data["modalityWeights"]["v"] > 0


def test_inference_with_audio():
    """Test inference with audio embedding"""
    request = {
        "ha": [0.1, 0.2, 0.3, 0.4, 0.5] * 7,  # 32 dims (approx)
        "meta": {
            "sessionId": "test-session",
            "consent": {
                "vision": False,
                "audio": True,
                "text": False,
                "context": False,
                "logs": False,
            },
            "timestamp": "2024-01-01T00:00:00Z",
        },
    }
    response = client.post("/infer", json=request)
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "modalityWeights" in data
    assert data["modalityWeights"]["a"] > 0


def test_inference_with_context():
    """Test inference with context embedding"""
    request = {
        "hc": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # 8 dims
        "meta": {
            "sessionId": "test-session",
            "consent": {
                "vision": False,
                "audio": False,
                "text": False,
                "context": True,
                "logs": False,
            },
            "timestamp": "2024-01-01T00:00:00Z",
        },
    }
    response = client.post("/infer", json=request)
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "modalityWeights" in data
    assert data["modalityWeights"]["c"] > 0


