"""
Comprehensive test suite for the MLOps Pipeline API.
"""
import pytest
import json
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

from src.api import app
from src.predict import ModelPredictor
from src.train import ModelTrainer

# Test client
client = TestClient(app)

# Test data
SAMPLE_DATA = {
    "feature1": 45.5,
    "feature2": 28.3,
    "feature3": 22.1
}

BATCH_DATA = [
    {"feature1": 45.5, "feature2": 28.3, "feature3": 22.1},
    {"feature1": 30.2, "feature2": 35.1, "feature3": 18.9},
    {"feature1": 55.8, "feature2": 25.4, "feature3": 31.2}
]

class TestAPIEndpoints:
    """Test API endpoints."""

    def test_home_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "MLOps Pipeline API" in data["message"]
        assert "version" in data
        assert "docs" in data

    def test_health_check(self):
        with patch('src.api.ModelPredictor') as mock_predictor_class:
            mock_predictor = MagicMock()
            mock_predictor.get_model_info.return_value = {
                "model_type": "logistic",
                "feature_columns": ["feature1", "feature2", "feature3"]
            }
            mock_predictor_class.return_value = mock_predictor

            with patch('src.api.predictor', mock_predictor):
                response = client.get("/health")
                data = response.json()
                assert response.status_code == 200
                assert data["status"] == "healthy"
                assert data["model_loaded"] is True
                assert "timestamp" in data

    def test_health_check_unhealthy(self):
        with patch('src.api.get_predictor') as mock_get_predictor:
            mock_get_predictor.side_effect = Exception("Model not found")
            response = client.get("/health")
            data = response.json()
            assert response.status_code == 200
            assert data["status"] == "unhealthy"
            assert data["model_loaded"] is False

    def test_predict_single(self):
        with patch('src.api.ModelPredictor') as mock_predictor_class:
            mock_predictor = MagicMock()
            mock_predictor.predict.return_value = {
                "prediction": 1,
                "probability": 0.85,
                "model_type": "logistic"
            }
            mock_predictor_class.return_value = mock_predictor

            with patch('src.api.predictor', mock_predictor):
                response = client.post("/predict", json=SAMPLE_DATA)
                data = response.json()
                assert response.status_code == 200
                assert data["prediction"] == 1
                assert data["probability"] == pytest.approx(0.85, rel=1e-2)
                assert data["model_type"] == "logistic"
                assert "timestamp" in data

    def test_predict_batch(self):
        with patch('src.api.ModelPredictor') as mock_predictor_class:
            mock_predictor = MagicMock()
            mock_predictor.predict.return_value = {
                "predictions": [1, 0, 1],
                "probabilities": [[0.2, 0.8], [0.7, 0.3], [0.1, 0.9]],
                "model_type": "logistic",
                "n_samples": 3
            }
            mock_predictor_class.return_value = mock_predictor

            with patch('src.api.predictor', mock_predictor):
                response = client.post("/predict/batch", json={"data": BATCH_DATA})
                data = response.json()
                assert response.status_code == 200
                assert len(data["predictions"]) == 3
                assert len(data["probabilities"]) == 3
                assert data["n_samples"] == 3
                assert "timestamp" in data

    def test_predict_validation_error(self):
        invalid_data = {"feature1": "not_a_number", "feature2": 28.3, "feature3": 22.1}
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422

    def test_predict_missing_features(self):
        incomplete_data = {"feature1": 45.5, "feature2": 28.3}
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422

    def test_model_info(self):
        with patch('src.api.ModelPredictor') as mock_predictor_class:
            mock_predictor = MagicMock()
            mock_predictor.get_model_info.return_value = {
                "model_type": "logistic",
                "feature_columns": ["feature1", "feature2", "feature3"],
                "training_date": "2024-01-01T00:00:00"
            }
            mock_predictor_class.return_value = mock_predictor

            with patch('src.api.predictor', mock_predictor):
                response = client.get("/model/info")
                data = response.json()
                assert response.status_code == 200
                assert data["model_type"] == "logistic"
                assert "feature_columns" in data

    def test_model_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir) / "metadata.json"
            metadata = {"train_accuracy": 0.95, "test_accuracy": 0.92, "test_auc": 0.89, "model_type": "logistic"}
            metadata_path.write_text(json.dumps(metadata))

            with patch('src.api.Path') as mock_path:
                mock_path.return_value.exists.return_value = True
                mock_path.return_value.__truediv__ = lambda self, other: Path(temp_dir) / other

                response = client.get("/model/metrics")
                data = response.json()
                assert response.status_code == 200
                assert "training_metrics" in data
                assert data["training_metrics"]["model_type"] == "logistic"

    def test_model_metrics_not_found(self):
        with patch('src.api.Path') as mock_path:
            mock_path.return_value.exists.return_value = False
            response = client.get("/model/metrics")
            data = response.json()
            assert response.status_code == 200
            assert "No training metrics available" in data["message"]


class TestModelPredictor:
    """Test ModelPredictor class."""

    def test_predictor_initialization(self):
        with patch('src.predict.joblib.load') as mock_load:
            mock_model, mock_scaler = MagicMock(), MagicMock()
            mock_load.side_effect = [mock_model, mock_scaler]
            with patch('src.predict.Path') as mock_path:
                mock_path.return_value.exists.return_value = True
                predictor = ModelPredictor()
                assert predictor.model is not None
                assert predictor.scaler is not None

    def test_predictor_fallback(self):
        with patch('src.predict.joblib.load') as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            with patch('src.predict.Path.exists', return_value=True):
                predictor = ModelPredictor()
                assert predictor.model is not None

    def test_validate_input_single(self):
        with patch('src.predict.joblib.load') as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            with patch('src.predict.Path') as mock_path:
                mock_path.return_value.exists.return_value = True
                predictor = ModelPredictor()
                predictor.feature_columns = ["feature1", "feature2", "feature3"]
                df = predictor.validate_input(SAMPLE_DATA)
                assert len(df) == 1
                assert list(df.columns) == ["feature1", "feature2", "feature3"]

    def test_validate_input_batch(self):
        with patch('src.predict.joblib.load') as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            with patch('src.predict.Path') as mock_path:
                mock_path.return_value.exists.return_value = True
                predictor = ModelPredictor()
                predictor.feature_columns = ["feature1", "feature2", "feature3"]
                df = predictor.validate_input(BATCH_DATA)
                assert len(df) == 3
                assert list(df.columns) == ["feature1", "feature2", "feature3"]

    def test_validate_input_missing_features(self):
        with patch('src.predict.joblib.load') as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model
            with patch('src.predict.Path') as mock_path:
                mock_path.return_value.exists.return_value = True
                predictor = ModelPredictor()
                predictor.feature_columns = ["feature1", "feature2", "feature3"]
                with pytest.raises(ValueError, match="Missing required features"):
                    predictor.validate_input({"feature1": 45.5, "feature2": 28.3})

    def test_predict_single(self):
        with patch('src.predict.joblib.load') as mock_load:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([1])
            mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
            mock_load.return_value = mock_model

            with patch('src.predict.Path') as mock_path:
                mock_path.return_value.exists.return_value = True
                predictor = ModelPredictor()
                predictor.feature_columns = ["feature1", "feature2", "feature3"]
                predictor.metadata = {"model_type": "logistic"}
                predictor.scaler = MagicMock()
                predictor.scaler.transform.return_value = np.array([[45.5, 28.3, 22.1]])
                result = predictor.predict(SAMPLE_DATA)
               
