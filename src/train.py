"""
Enhanced model training script with comprehensive evaluation and logging.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import json
import os
from datetime import datetime
import logging
from data_loader import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training, evaluation, and persistence."""

    def __init__(self, model_type: str = "logistic", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.data_loader = DataLoader(random_state=random_state)
        self.training_metrics = {}

    def _get_model(self):
        """Get model instance based on model_type."""
        if self.model_type == "logistic":
            return LogisticRegression(random_state=self.random_state, max_iter=1000)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, max_depth=10
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train_model(self, data_path: str = None, test_size: float = 0.2):
        """
        Train the model with comprehensive evaluation.

        Args:
            data_path: Path to training data (optional)
            test_size: Proportion of data for testing
        """
        logger.info("Starting model training...")

        # Load and preprocess data
        df = self.data_loader.load_data(data_path)
        df = self.data_loader.preprocess_data(df)

        # Prepare features and target
        X, y = self.data_loader.prepare_features_target(df)

        # Split data
        X_train, X_test, y_train, y_test = self.data_loader.split_data(X, y, test_size)

        # Initialize and train model
        self.model = self._get_model()
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)

        # Evaluate model
        self._evaluate_model(X_train, X_test, y_train, y_test)

        # Save model and metadata
        self._save_model()

        logger.info("Model training completed successfully!")

    def _evaluate_model(self, X_train, X_test, y_train, y_test):
        """Evaluate model performance on train and test sets."""
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        y_test_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        test_auc = roc_auc_score(y_test, y_test_pred_proba)

        # Store metrics
        self.training_metrics = {
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "test_auc": float(test_auc),
            "model_type": self.model_type,
            "feature_columns": self.data_loader.feature_columns,
            "training_date": datetime.now().isoformat(),
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
        }

        # Log metrics
        logger.info(f"Training Accuracy: {train_accuracy:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test AUC: {test_auc:.4f}")

        # Print detailed classification report
        logger.info("\nClassification Report (Test Set):")
        logger.info(f"\n{classification_report(y_test, y_test_pred)}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        logger.info(f"\nConfusion Matrix:\n{cm}")

    def _save_model(self):
        """Save model and metadata."""
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)

        # Save model
        model_path = "models/model.pkl"
        joblib.dump(self.model, model_path)
        logger.info(f"Model saved to {model_path}")

        # Save scaler
        scaler_path = "models/scaler.pkl"
        joblib.dump(self.data_loader.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

        # Save metadata
        metadata_path = "models/metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.training_metrics, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")

        # Also save to src/ for backward compatibility
        joblib.dump(self.model, "src/model.pkl")
        logger.info("Model also saved to src/model.pkl for backward compatibility")


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument(
        "--model",
        choices=["logistic", "random_forest"],
        default="logistic",
        help="Model type to train",
    )
    parser.add_argument("--data", type=str, help="Path to training data")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Proportion of data for testing"
    )

    args = parser.parse_args()

    # Train model
    trainer = ModelTrainer(model_type=args.model, random_state=42)
    trainer.train_model(data_path=args.data, test_size=args.test_size)


if __name__ == "__main__":
    main()
