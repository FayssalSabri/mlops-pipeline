"""
Unit tests for the DataLoader class.
"""
import pytest
import pandas as pd
import numpy as np
from src.data_loader import DataLoader


class TestDataLoader:
    """Test DataLoader functionality."""
    
    def test_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader(random_state=42)
        assert loader.random_state == 42
        assert loader.scaler is not None
        assert loader.feature_columns is None
    
    def test_generate_sample_data(self):
        """Test sample data generation."""
        loader = DataLoader(random_state=42)
        df = loader.generate_sample_data(n_samples=100)
        
        # Check shape
        assert len(df) == 100
        assert df.shape[1] == 4  # 3 features + target
        
        # Check columns
        expected_columns = ['feature1', 'feature2', 'feature3', 'target']
        assert list(df.columns) == expected_columns
        
        # Check data types
        assert df['target'].dtype in [np.int64, np.int32]
        assert df['target'].isin([0, 1]).all()
        
        # Check feature ranges (should be reasonable)
        for col in ['feature1', 'feature2', 'feature3']:
            assert df[col].min() > 0  # All positive
            assert df[col].max() < 100  # Reasonable upper bound
    
    def test_generate_sample_data_different_sizes(self):
        """Test sample data generation with different sizes."""
        loader = DataLoader(random_state=42)
        
        for n_samples in [10, 50, 100, 500]:
            df = loader.generate_sample_data(n_samples=n_samples)
            assert len(df) == n_samples
    
    def test_load_data_with_file(self):
        """Test loading data from file."""
        loader = DataLoader(random_state=42)
        
        # Create temporary CSV file
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 3, 4, 5, 6],
            'feature3': [3, 4, 5, 6, 7],
            'target': [0, 1, 0, 1, 0]
        })
        
        with pytest.raises(FileNotFoundError):
            # This should raise an error since file doesn't exist
            loader.load_data("nonexistent_file.csv")
    
    def test_load_data_generate_sample(self):
        """Test loading data when no file is provided."""
        loader = DataLoader(random_state=42)
        df = loader.load_data()
        
        # Should generate sample data
        assert len(df) == 1000  # Default n_samples
        assert 'target' in df.columns
    
    def test_preprocess_data_no_missing_values(self):
        """Test preprocessing data without missing values."""
        loader = DataLoader(random_state=42)
        
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 3, 4, 5, 6],
            'feature3': [3, 4, 5, 6, 7],
            'target': [0, 1, 0, 1, 0]
        })
        
        processed_df = loader.preprocess_data(df)
        assert len(processed_df) == len(df)  # No outliers to remove
        assert list(processed_df.columns) == list(df.columns)
    
    def test_preprocess_data_with_missing_values(self):
        """Test preprocessing data with missing values."""
        loader = DataLoader(random_state=42)
        
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [2, 3, 4, np.nan, 6],
            'feature3': [3, 4, 5, 6, 7],
            'target': [0, 1, 0, 1, 0]
        })
        
        processed_df = loader.preprocess_data(df)
        assert processed_df.isnull().sum().sum() == 0  # No missing values
        assert len(processed_df) == len(df)  # Same length
    
    def test_preprocess_data_with_outliers(self):
        """Test preprocessing data with outliers."""
        loader = DataLoader(random_state=42)
        
        # Create data with clear outliers
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 1000],  # 1000 is an outlier
            'feature2': [2, 3, 4, 5, 6, 7],
            'feature3': [3, 4, 5, 6, 7, 8],
            'target': [0, 1, 0, 1, 0, 1]
        })
        
        processed_df = loader.preprocess_data(df)
        assert len(processed_df) < len(df)  # Outliers should be removed
        assert processed_df['feature1'].max() < 1000  # Outlier removed
    
    def test_prepare_features_target(self):
        """Test preparing features and target."""
        loader = DataLoader(random_state=42)
        
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 3, 4, 5, 6],
            'feature3': [3, 4, 5, 6, 7],
            'target': [0, 1, 0, 1, 0]
        })
        
        X, y = loader.prepare_features_target(df)
        
        # Check features
        assert X.shape[0] == 5
        assert X.shape[1] == 3
        assert list(X.columns) == ['feature1', 'feature2', 'feature3']
        
        # Check target
        assert len(y) == 5
        assert y.name == 'target'
        assert y.dtype in [np.int64, np.int32]
        
        # Check scaler was fitted
        assert loader.feature_columns == ['feature1', 'feature2', 'feature3']
        assert hasattr(loader.scaler, 'mean_')  # Scaler was fitted
    
    def test_prepare_features_target_custom_column(self):
        """Test preparing features and target with custom target column."""
        loader = DataLoader(random_state=42)
        
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 3, 4, 5, 6],
            'feature3': [3, 4, 5, 6, 7],
            'label': [0, 1, 0, 1, 0]  # Different target column name
        })
        
        X, y = loader.prepare_features_target(df, target_column='label')
        
        assert X.shape[1] == 3  # 3 features
        assert len(y) == 5
        assert y.name == 'label'
    
    def test_prepare_features_target_missing_target(self):
        """Test preparing features and target with missing target column."""
        loader = DataLoader(random_state=42)
        
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 3, 4, 5, 6],
            'feature3': [3, 4, 5, 6, 7]
            # No target column
        })
        
        with pytest.raises(ValueError, match="Target column 'target' not found"):
            loader.prepare_features_target(df)
    
    def test_split_data(self):
        """Test data splitting."""
        loader = DataLoader(random_state=42)
        
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            'feature3': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        })
        y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        
        X_train, X_test, y_train, y_test = loader.split_data(X, y, test_size=0.2)
        
        # Check sizes
        assert len(X_train) == 8
        assert len(X_test) == 2
        assert len(y_train) == 8
        assert len(y_test) == 2
        
        # Check no overlap
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        assert len(train_indices.intersection(test_indices)) == 0
    
    def test_split_data_different_test_sizes(self):
        """Test data splitting with different test sizes."""
        loader = DataLoader(random_state=42)
        
        X = pd.DataFrame({
            'feature1': list(range(20)),
            'feature2': list(range(20)),
            'feature3': list(range(20))
        })
        y = pd.Series([i % 2 for i in range(20)])
        
        for test_size in [0.1, 0.3, 0.5]:
            X_train, X_test, y_train, y_test = loader.split_data(X, y, test_size=test_size)
            
            expected_test_size = int(len(X) * test_size)
            expected_train_size = len(X) - expected_test_size
            
            assert len(X_test) == expected_test_size
            assert len(X_train) == expected_train_size
    
    def test_transform_features(self):
        """Test feature transformation."""
        loader = DataLoader(random_state=42)
        
        # First prepare features to fit scaler
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 3, 4, 5, 6],
            'feature3': [3, 4, 5, 6, 7],
            'target': [0, 1, 0, 1, 0]
        })
        
        X, y = loader.prepare_features_target(df)
        
        # Now test transformation
        new_data = pd.DataFrame({
            'feature1': [10, 20],
            'feature2': [30, 40],
            'feature3': [50, 60]
        })
        
        transformed = loader.transform_features(new_data)
        
        assert transformed.shape == new_data.shape
        assert list(transformed.columns) == list(new_data.columns)
        assert list(transformed.index) == list(new_data.index)
    
    def test_transform_features_not_fitted(self):
        """Test feature transformation when scaler is not fitted."""
        loader = DataLoader(random_state=42)
        
        new_data = pd.DataFrame({
            'feature1': [10, 20],
            'feature2': [30, 40],
            'feature3': [50, 60]
        })
        
        with pytest.raises(ValueError, match="Scaler not fitted"):
            loader.transform_features(new_data)
    
    def test_reproducibility(self):
        """Test that data generation is reproducible with same random state."""
        loader1 = DataLoader(random_state=42)
        loader2 = DataLoader(random_state=42)
        
        df1 = loader1.generate_sample_data(n_samples=100)
        df2 = loader2.generate_sample_data(n_samples=100)
        
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_different_random_states(self):
        """Test that different random states produce different data."""
        loader1 = DataLoader(random_state=42)
        loader2 = DataLoader(random_state=123)
        
        df1 = loader1.generate_sample_data(n_samples=100)
        df2 = loader2.generate_sample_data(n_samples=100)
        
        # Data should be different
        assert not df1.equals(df2)
        
        # But should have same structure
        assert df1.shape == df2.shape
        assert list(df1.columns) == list(df2.columns)
