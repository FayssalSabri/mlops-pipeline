"""
Data loading and preprocessing utilities for the MLOps pipeline.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles data loading, preprocessing, and validation."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_columns = None

    def generate_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate a more realistic sample dataset for demonstration.

        Args:
            n_samples: Number of samples to generate

        Returns:
            DataFrame with features and target
        """
        np.random.seed(self.random_state)

        # Generate features with some correlation
        feature1 = np.random.normal(50, 15, n_samples)
        feature2 = np.random.normal(30, 10, n_samples)
        feature3 = np.random.normal(25, 8, n_samples)

        # Create target with some logical relationship
        # Higher values of feature1 and feature2 increase probability of target=1
        target_prob = 1 / (
            1 + np.exp(-(0.1 * feature1 + 0.15 * feature2 - 0.05 * feature3 - 5))
        )
        target = np.random.binomial(1, target_prob, n_samples)

        df = pd.DataFrame(
            {
                "feature1": feature1,
                "feature2": feature2,
                "feature3": feature3,
                "target": target,
            }
        )

        logger.info(f"Generated dataset with {n_samples} samples")
        return df

    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from file or generate sample data.

        Args:
            file_path: Path to data file (CSV format)

        Returns:
            Loaded DataFrame
        """
        if file_path and pd.io.common.file_exists(file_path):
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data from {file_path}")
        else:
            logger.info("No data file found, generating sample data")
            df = self.generate_sample_data()

        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data: handle missing values, outliers, etc.

        Args:
            df: Input DataFrame

        Returns:
            Preprocessed DataFrame
        """
        # Handle missing values
        if df.isnull().sum().sum() > 0:
            logger.warning("Found missing values, filling with median")
            df = df.fillna(df.median())

        # Remove outliers using IQR method
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != "target":  # Don't remove outliers from target
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        logger.info(f"Preprocessed data shape: {df.shape}")
        return df

    def prepare_features_target(
        self, df: pd.DataFrame, target_column: str = "target"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target, and fit scaler.

        Args:
            df: Input DataFrame
            target_column: Name of target column

        Returns:
            Tuple of (features, target)
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        # Store feature columns for later use
        self.feature_columns = [col for col in df.columns if col != target_column]

        X = df[self.feature_columns]
        y = df[target_column]

        # Fit scaler on features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), columns=self.feature_columns, index=X.index
        )

        logger.info(f"Prepared {len(self.feature_columns)} features")
        return X_scaled, y

    def split_data(
        self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
    ) -> Tuple:
        """
        Split data into train and test sets.

        Args:
            X: Features
            y: Target
            test_size: Proportion of data for testing

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        logger.info(f"Split data: train={X_train.shape[0]}, test={X_test.shape[0]}")
        return X_train, X_test, y_train, y_test

    def transform_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scaler.

        Args:
            X: Features to transform

        Returns:
            Transformed features
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call prepare_features_target first.")

        return pd.DataFrame(
            self.scaler.transform(X), columns=self.feature_columns, index=X.index
        )
