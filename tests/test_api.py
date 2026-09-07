import pytest
from fastapi.testclient import TestClient
from main import app

VALID_PAYLOAD = {
    "Marital status": 1,
    "Application mode": 1,
    "Application order": 1,
    "Course": "9500",
    "Daytime/evening attendance\t": 1,
    "Previous qualification": 1,
    "Previous qualification (grade)": 150.0,
    "Nacionality": 1,
    "Mother's qualification": 2,
    "Father's qualification": 2,
    "Mother's occupation": 3,
    "Father's occupation": 3,
    "Admission grade": 140.5,
    "Displaced": 0,
    "Educational special needs": 0,
    "Debtor": 0,
    "Tuition fees up to date": 1,
    "Gender": 1,
    "Scholarship holder": 0,
    "Age at enrollment": 20,
    "International": 0,
    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (enrolled)": 6,
    "Curricular units 1st sem (evaluations)": 8,
    "Curricular units 1st sem (approved)": 5,
    "Curricular units 1st sem (grade)": 13.5,
    "Curricular units 1st sem (without evaluations)": 0,
    "Curricular units 2nd sem (credited)": 0,
    "Curricular units 2nd sem (enrolled)": 6,
    "Curricular units 2nd sem (evaluations)": 7,
    "Curricular units 2nd sem (approved)": 4,
    "Curricular units 2nd sem (grade)": 12.8,
    "Curricular units 2nd sem (without evaluations)": 0,
    "Unemployment rate": 10.8,
    "Inflation rate": 1.4,
    "GDP": 1.74,
}

VALID_TARGET_LABELS = {"Graduate", "Dropout", "Enrolled"}


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as c:
        yield c


def test_predict_valid_input_returns_200(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_predict_response_has_expected_shape(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    body = response.json()

    assert "prediction" in body
    assert "probabilities" in body
    assert body["prediction"] in VALID_TARGET_LABELS
    assert set(body["probabilities"].keys()) == VALID_TARGET_LABELS


def test_predict_probabilities_sum_to_one(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    probabilities = response.json()["probabilities"]
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=0.01)


def test_predict_missing_field_returns_422(client):
    incomplete_payload = dict(VALID_PAYLOAD)
    del incomplete_payload["GDP"]

    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422


def test_predict_wrong_type_returns_422(client):
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["Age at enrollment"] = "not_a_number"

    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_unknown_category_does_not_crash(client):
    """The model should be able to handle unknown categories."""
    payload = dict(VALID_PAYLOAD)
    payload["Course"] = "totally_unseen_course_code"

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] in VALID_TARGET_LABELS
