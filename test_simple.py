"""
Simple test to verify basic functionality.
"""
import pytest
from fastapi.testclient import TestClient
from src.api import app

def test_home_endpoint():
    """Test home endpoint."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "MLOps Pipeline API" in data["message"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
