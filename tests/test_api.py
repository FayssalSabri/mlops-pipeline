from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "API MLOps" in response.json()["message"]
