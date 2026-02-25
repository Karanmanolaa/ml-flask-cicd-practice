import pytest
from app import app

@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client

def valid_payload():
    return {
        "job": "admin.",
        "marital": "single",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "poutcome": "nonexistent",
        "age": 35,
        "duration": 200,
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "emp.var.rate": 1.1,
        "cons.price.idx": 93.2,
        "cons.conf.idx": -40.0
    }

def test_predict_api_success(client):
    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200

    data = response.get_json()

    assert "prediction" in data
    assert "probability" in data
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probability"], float)