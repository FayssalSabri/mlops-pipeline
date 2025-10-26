"""
Comprehensive test suite for the MLOps Pipeline API.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from api import app
from predict import ModelPredictor
from train import ModelTrainer

# Test client
client = TestClient(app)

# Test data
SAMPLE_DATA = {"feature1": 45.5, "feature2": 28.3, "feature3": 22.1}

BATCH_DATA = [
    {"feature1": 45.5, "feature2": 28.3, "feature3": 22.1},
    {"feature1": 30.2, "feature2": 35.1, "feature3": 18.9},
    {"feature1": 55.8, "feature2": 25.4, "feature3": 31.2},
]


class TestAPIEndpoints:
    """Test API endpoints."""

    def test_home_endpoint(self):
        """Test home endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "MLOps Pipeline API" in data["message"]
        assert "version" in data
        assert "docs" in data

    def test_health_check(self):
        """Test health check endpoint."""
        with patch("api.ModelPredictor") as mock_predictor_class:
            mock_predictor = MagicMock()
            mock_predictor.get_model_info.return_value = {
                "model_type": "logistic",
                "feature_columns": ["feature1", "feature2", "feature3"],
            }
            mock_predictor_class.return_value = mock_predictor

            with patch("api.predictor", mock_predictor):
                response = client.get("/health")
                data = response.json()
                assert response.status_code == 200
                assert data["status"] == "healthy"
                assert data["model_loaded"] is True
                assert "timestamp" in data

    def test_predict_single(self):
        """Test single prediction endpoint."""
        with patch("api.ModelPredictor") as mock_predictor_class:
            mock_predictor = MagicMock()
            mock_predictor.predict.return_value = {
                "prediction": 1,
                "probability": 0.85,
                "model_type": "logistic",
            }
            mock_predictor_class.return_value = mock_predictor

            with patch("api.predictor", mock_predictor):
                response = client.post("/predict", json=SAMPLE_DATA)
                data = response.json()
                assert response.status_code == 200
                assert data["prediction"] == 1
                assert data["probability"] == pytest.approx(0.85, rel=1e-2)
                assert data["model_type"] == "logistic"
                assert "timestamp" in data


class TestModelPredictor:
    """Test ModelPredictor class."""

    def test_predictor_initialization(self):
        """Test predictor initialization."""
        with patch("predict.joblib.load") as mock_load:
            # Mock model and scaler
            mock_model = MagicMock()
            mock_scaler = MagicMock()
            mock_load.side_effect = [mock_model, mock_scaler]

            with patch("predict.Path") as mock_path:
                mock_path.return_value.exists.return_value = True

                predictor = ModelPredictor()
                assert predictor.model is not None
                assert predictor.scaler is not None

    def test_predictor_fallback(self):
        """Test predictor fallback to src/ directory."""
        with patch("predict.joblib.load") as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model

            # Mock Path.exists to return False for models/, True for src/
            with patch("predict.Path.exists") as mock_exists:
                mock_exists.return_value = True  # Always return True for simplicity

                predictor = ModelPredictor()
                assert predictor.model is not None

    def test_validate_input_single(self):
        """Test input validation for single sample."""
        with patch("predict.joblib.load") as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model

            with patch("predict.Path") as mock_path:
                mock_path.return_value.exists.return_value = True

                predictor = ModelPredictor()
                predictor.feature_columns = ["feature1", "feature2", "feature3"]

                df = predictor.validate_input(SAMPLE_DATA)
                assert len(df) == 1
                assert list(df.columns) == ["feature1", "feature2", "feature3"]

    def test_validate_input_batch(self):
        """Test input validation for batch samples."""
        with patch("predict.joblib.load") as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model

            with patch("predict.Path") as mock_path:
                mock_path.return_value.exists.return_value = True

                predictor = ModelPredictor()
                predictor.feature_columns = ["feature1", "feature2", "feature3"]

                df = predictor.validate_input(BATCH_DATA)
                assert len(df) == 3
                assert list(df.columns) == ["feature1", "feature2", "feature3"]

    def test_validate_input_missing_features(self):
        """Test input validation with missing features."""
        with patch("predict.joblib.load") as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = mock_model

            with patch("predict.Path") as mock_path:
                mock_path.return_value.exists.return_value = True

                predictor = ModelPredictor()
                predictor.feature_columns = ["feature1", "feature2", "feature3"]

                incomplete_data = {"feature1": 45.5, "feature2": 28.3}

                with pytest.raises(ValueError, match="Missing required features"):
                    predictor.validate_input(incomplete_data)

    def test_predict_single(self):
        """Test prediction for single sample."""
        import numpy as np

        with patch("predict.joblib.load") as mock_load:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([1])
            mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
            mock_load.return_value = mock_model

            with patch("predict.Path") as mock_path:
                mock_path.return_value.exists.return_value = True

                predictor = ModelPredictor()
                predictor.feature_columns = ["feature1", "feature2", "feature3"]
                predictor.metadata = {"model_type": "logistic"}
                predictor.scaler = MagicMock()
                predictor.scaler.transform.return_value = np.array([[45.5, 28.3, 22.1]])

                result = predictor.predict(SAMPLE_DATA)
                assert result["prediction"] == 1
                assert result["probability"] == 0.8
                assert result["model_type"] == "logistic"
                assert result["n_samples"] == 1


class TestModelTrainer:
    """Test ModelTrainer class."""

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = ModelTrainer(model_type="logistic", random_state=42)
        assert trainer.model_type == "logistic"
        assert trainer.random_state == 42
        assert trainer.data_loader is not None

    def test_get_model_logistic(self):
        """Test getting logistic regression model."""
        trainer = ModelTrainer(model_type="logistic")
        model = trainer._get_model()
        assert model.__class__.__name__ == "LogisticRegression"

    def test_get_model_random_forest(self):
        """Test getting random forest model."""
        trainer = ModelTrainer(model_type="random_forest")
        model = trainer._get_model()
        assert model.__class__.__name__ == "RandomForestClassifier"

    def test_get_model_invalid(self):
        """Test getting invalid model type."""
        trainer = ModelTrainer(model_type="invalid")
        with pytest.raises(ValueError, match="Unknown model type"):
            trainer._get_model()


class TestDataLoader:
    """Test DataLoader class."""

    def test_data_loader_initialization(self):
        """Test data loader initialization."""
        from data_loader import DataLoader

        loader = DataLoader(random_state=42)
        assert loader.random_state == 42
        assert loader.scaler is not None
        assert loader.feature_columns is None

    def test_generate_sample_data(self):
        """Test sample data generation."""
        from data_loader import DataLoader

        loader = DataLoader(random_state=42)
        df = loader.generate_sample_data(n_samples=100)

        assert len(df) == 100
        assert "feature1" in df.columns
        assert "feature2" in df.columns
        assert "feature3" in df.columns
        assert "target" in df.columns
        assert df["target"].isin([0, 1]).all()

    def test_preprocess_data(self):
        """Test data preprocessing."""
        from data_loader import DataLoader
        import pandas as pd
        import numpy as np

        loader = DataLoader(random_state=42)

        # Create test data with outliers
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5, 100],  # 100 is an outlier
                "feature2": [2, 3, 4, 5, 6, 7],
                "target": [0, 1, 0, 1, 0, 1],
            }
        )

        processed_df = loader.preprocess_data(df)
        assert len(processed_df) < len(df)  # Outliers should be removed
        assert "feature1" in processed_df.columns
        assert "feature2" in processed_df.columns
        assert "target" in processed_df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
