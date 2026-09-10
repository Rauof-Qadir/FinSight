from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_root():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "online"


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    assert data["model"] == "XGBoost"


def test_model_info():

    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "XGBoost"

    assert "features" in data