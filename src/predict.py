"""
Enhanced prediction script with proper data validation and error handling.
"""
import pandas as pd
import numpy as np
import joblib
import json
import logging
from typing import Dict, List, Union, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPredictor:
    """Handles model loading and prediction with validation."""
    
    def __init__(self, model_path: str = "models/model.pkl", 
                 scaler_path: str = "models/scaler.pkl",
                 metadata_path: str = "models/metadata.json"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.metadata_path = metadata_path
        self.model = None
        self.scaler = None
        self.metadata = None
        self.feature_columns = None
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model, scaler, and metadata."""
        try:
            # Load model
            if Path(self.model_path).exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")
            else:
                # Fallback to src/ directory
                fallback_path = "src/model.pkl"
                if Path(fallback_path).exists():
                    self.model = joblib.load(fallback_path)
                    logger.info(f"Model loaded from {fallback_path}")
                else:
                    raise FileNotFoundError(f"Model not found at {self.model_path} or {fallback_path}")
            
            # Load scaler
            if Path(self.scaler_path).exists():
                self.scaler = joblib.load(self.scaler_path)
                logger.info(f"Scaler loaded from {self.scaler_path}")
            else:
                logger.warning("Scaler not found, predictions may be inaccurate")
            
            # Load metadata
            if Path(self.metadata_path).exists():
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                self.feature_columns = self.metadata.get('feature_columns', [])
                logger.info(f"Metadata loaded from {self.metadata_path}")
            else:
                logger.warning("Metadata not found, using default feature columns")
                # Default feature columns for backward compatibility
                self.feature_columns = ['feature1', 'feature2', 'feature3']
                
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise
    
    def validate_input(self, data: Union[Dict, List[Dict]]) -> pd.DataFrame:
        """
        Validate and prepare input data for prediction.
        
        Args:
            data: Input data as dict or list of dicts
            
        Returns:
            Validated DataFrame
        """
        if isinstance(data, dict):
            data = [data]
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Check if all required features are present
        missing_features = set(self.feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select only the required features in the correct order
        df = df[self.feature_columns]
        
        # Check for missing values
        if df.isnull().any().any():
            raise ValueError("Input data contains missing values")
        
        # Check data types
        for col in self.feature_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    raise ValueError(f"Feature '{col}' must be numeric")
        
        logger.info(f"Input validation successful: {df.shape[0]} samples")
        return df
    
    def predict(self, data: Union[Dict, List[Dict]]) -> Dict:
        """
        Make predictions on input data.
        
        Args:
            data: Input data as dict or list of dicts
            
        Returns:
            Dictionary with predictions and probabilities
        """
        try:
            # Validate input
            df = self.validate_input(data)
            
            # Scale features if scaler is available
            if self.scaler is not None:
                df_scaled = pd.DataFrame(
                    self.scaler.transform(df),
                    columns=self.feature_columns,
                    index=df.index
                )
            else:
                df_scaled = df
                logger.warning("No scaler available, using unscaled features")
            
            # Make predictions
            predictions = self.model.predict(df_scaled)
            probabilities = self.model.predict_proba(df_scaled)
            
            # Prepare results
            results = {
                'predictions': predictions.tolist(),
                'probabilities': probabilities.tolist(),
                'model_type': self.metadata.get('model_type', 'unknown') if self.metadata else 'unknown',
                'n_samples': len(df)
            }
            
            # Add individual predictions if single sample
            if len(df) == 1:
                results['prediction'] = int(predictions[0])
                results['probability'] = float(probabilities[0][1])  # Probability of positive class
            
            logger.info(f"Predictions completed for {len(df)} samples")
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.metadata:
            return self.metadata
        else:
            return {
                'model_type': 'unknown',
                'feature_columns': self.feature_columns,
                'status': 'metadata not available'
            }


def predict(data: Union[Dict, List[Dict]], 
           model_path: str = "models/model.pkl") -> Dict:
    """
    Convenience function for making predictions.
    
    Args:
        data: Input data as dict or list of dicts
        model_path: Path to model file
        
    Returns:
        Dictionary with predictions
    """
    predictor = ModelPredictor(model_path=model_path)
    return predictor.predict(data)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Make predictions')
    parser.add_argument('--data', type=str, help='JSON string with input data')
    parser.add_argument('--model', type=str, default='models/model.pkl', 
                       help='Path to model file')
    
    args = parser.parse_args()
    
    if args.data:
        import json
        data = json.loads(args.data)
        result = predict(data, args.model)
        print(json.dumps(result, indent=2))
    else:
        # Default example
        example_data = {
            'feature1': 45.5,
            'feature2': 28.3,
            'feature3': 22.1
        }
        result = predict(example_data, args.model)
        print("Example prediction:")
        print(json.dumps(result, indent=2))