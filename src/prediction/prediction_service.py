"""
Prediction Service for hotel cancellation predictions.

This module provides the main prediction service that loads trained models
and preprocessors, and provides methods for making predictions on new bookings.
"""

import pickle
import time
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from src.utils.logger import get_logger
from src.utils.exceptions import PredictionError
from src.data_processing.feature_engineer import FeatureEngineer

logger = get_logger(__name__)


class PredictionService:
    """
    Service for making cancellation predictions on hotel bookings.
    
    This class handles loading model artifacts, preprocessing new data,
    and generating predictions with probabilities and risk levels.
    """
    
    def __init__(
        self,
        model_path: str = "models/best_model.pkl",
        preprocessor_path: Optional[str] = None,
        feature_engineer_dir: Optional[str] = "data/processed"
    ):
        """
        Initialize the PredictionService.
        
        Args:
            model_path: Path to the saved model file
            preprocessor_path: Path to the saved preprocessor file (optional, deprecated)
            feature_engineer_dir: Directory containing feature engineer transformers
        
        Raises:
            PredictionError: If initialization fails
        """
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path) if preprocessor_path else None
        self.feature_engineer_dir = Path(feature_engineer_dir) if feature_engineer_dir else None
        
        self.model = None
        self.preprocessor = None
        self.feature_engineer = None
        self.metadata = {}
        self.feature_names = []
        
        logger.info(f"Initializing PredictionService with model: {model_path}")
        
        # Load artifacts on initialization
        self.load_artifacts()
    
    def load_artifacts(self) -> None:
        """
        Load model and preprocessor artifacts from disk.
        
        This method loads the trained model and any preprocessing artifacts
        needed for making predictions on new data.
        
        Raises:
            PredictionError: If loading fails
        """
        try:
            # Load model
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            logger.info(f"Loading model from: {self.model_path}")
            
            with open(self.model_path, 'rb') as f:
                model_package = pickle.load(f)
            
            # Extract model and metadata
            if isinstance(model_package, dict):
                self.model = model_package.get("model")
                self.metadata = model_package.get("metadata", {})
                
                if self.model is None:
                    raise ValueError("Model package does not contain 'model' key")
            else:
                # Handle legacy models without metadata
                self.model = model_package
                self.metadata = {}
                logger.warning("Loaded model without metadata structure")
            
            logger.info(
                f"Model loaded successfully: {self.metadata.get('model_type', 'Unknown')} "
                f"(version {self.metadata.get('version', 'unknown')})"
            )
            
            # Load feature engineer if directory is provided
            if self.feature_engineer_dir and self.feature_engineer_dir.exists():
                logger.info(f"Loading feature engineer from: {self.feature_engineer_dir}")
                
                try:
                    self.feature_engineer = FeatureEngineer()
                    self.feature_engineer.load_transformers(
                        input_dir=str(self.feature_engineer_dir),
                        prefix="feature_engineer"
                    )
                    self.feature_names = self.feature_engineer.get_feature_names()
                    logger.info(f"Feature engineer loaded successfully with {len(self.feature_names)} features")
                except Exception as e:
                    logger.warning(f"Failed to load feature engineer: {str(e)}")
                    self.feature_engineer = None
            
            # Fallback to legacy preprocessor if feature engineer not loaded
            if self.feature_engineer is None:
                if self.preprocessor_path and self.preprocessor_path.exists():
                    logger.info(f"Loading preprocessor from: {self.preprocessor_path}")
                    
                    with open(self.preprocessor_path, 'rb') as f:
                        preprocessor_package = pickle.load(f)
                    
                    if isinstance(preprocessor_package, dict):
                        self.preprocessor = preprocessor_package.get("preprocessor")
                        self.feature_names = preprocessor_package.get("feature_names", [])
                    else:
                        self.preprocessor = preprocessor_package
                    
                    logger.info("Preprocessor loaded successfully")
                else:
                    logger.warning("No feature engineer or preprocessor available - predictions may fail with raw data")
            
        except FileNotFoundError as e:
            error_msg = f"Artifact file not found: {str(e)}"
            logger.error(error_msg)
            raise PredictionError(error_msg)
        
        except Exception as e:
            error_msg = f"Failed to load artifacts: {str(e)}"
            logger.error(error_msg)
            raise PredictionError(error_msg)
    
    def predict(self, booking_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a cancellation prediction for a single booking.
        
        Args:
            booking_data: Dictionary containing booking features
        
        Returns:
            Dict containing:
                - prediction: Binary prediction (0 = no cancellation, 1 = cancellation)
                - probability: Probability of cancellation (0.0 to 1.0)
                - risk_level: Risk category ('low', 'medium', 'high')
                - confidence: Model confidence score
                - timestamp: Prediction timestamp
        
        Raises:
            PredictionError: If prediction fails
        """
        try:
            start_time = time.time()
            
            # Get probability
            probability = self.predict_proba(booking_data)
            
            # Make binary prediction using default threshold of 0.5
            prediction = 1 if probability >= 0.5 else 0
            
            # Determine risk level
            if probability < 0.3:
                risk_level = "low"
            elif probability < 0.7:
                risk_level = "medium"
            else:
                risk_level = "high"
            
            # Calculate confidence (distance from decision boundary)
            confidence = abs(probability - 0.5) * 2  # Scale to 0-1
            
            # Calculate prediction time
            prediction_time = time.time() - start_time
            
            result = {
                "prediction": prediction,
                "probability": float(probability),
                "risk_level": risk_level,
                "confidence": float(confidence),
                "timestamp": datetime.now().isoformat(),
                "prediction_time_ms": round(prediction_time * 1000, 2)
            }
            
            logger.debug(
                f"Prediction made: {prediction} (probability: {probability:.3f}, "
                f"risk: {risk_level}, time: {prediction_time*1000:.2f}ms)"
            )
            
            return result
        
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            logger.error(error_msg)
            raise PredictionError(error_msg)
    
    def predict_proba(self, booking_data: Dict[str, Any]) -> float:
        """
        Get the cancellation probability for a single booking.
        
        Args:
            booking_data: Dictionary containing booking features
        
        Returns:
            float: Probability of cancellation (0.0 to 1.0)
        
        Raises:
            PredictionError: If prediction fails
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Call load_artifacts() first.")
            
            # Preprocess the data using feature engineer
            if self.feature_engineer is not None:
                # Convert dict to DataFrame for feature engineer
                df = pd.DataFrame([booking_data])
                
                # Apply feature engineering transformations
                X = self.feature_engineer.transform(df)
                
                logger.debug(f"Transformed data shape: {X.shape}")
            elif self.preprocessor is not None:
                # Legacy preprocessor path
                X = self.preprocessor.transform([booking_data])
            else:
                # If no preprocessor, assume booking_data is already in correct format
                # Convert dict to list of values in correct order
                if self.feature_names:
                    X = [[booking_data.get(feat, 0) for feat in self.feature_names]]
                else:
                    # Try to use the data as-is
                    import numpy as np
                    if isinstance(booking_data, dict):
                        X = [list(booking_data.values())]
                    else:
                        X = [booking_data]
            
            # Get probability prediction
            if hasattr(self.model, 'predict_proba'):
                # For classifiers with predict_proba method
                proba = self.model.predict_proba(X)
                # Return probability of positive class (cancellation)
                cancellation_probability = proba[0][1]
            elif hasattr(self.model, 'predict'):
                # Fallback to predict if predict_proba not available
                prediction = self.model.predict(X)
                cancellation_probability = float(prediction[0])
            else:
                raise ValueError("Model does not have predict or predict_proba method")
            
            return float(cancellation_probability)
        
        except Exception as e:
            error_msg = f"Probability prediction failed: {str(e)}"
            logger.error(error_msg)
            raise PredictionError(error_msg)
    
    def predict_batch(
        self,
        bookings: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple bookings at once.
        
        Args:
            bookings: List of booking data dictionaries
        
        Returns:
            List of prediction result dictionaries
        
        Raises:
            PredictionError: If batch prediction fails
        """
        try:
            if not bookings:
                logger.warning("Empty bookings list provided")
                return []
            
            logger.info(f"Making batch predictions for {len(bookings)} bookings")
            start_time = time.time()
            
            results = []
            
            for i, booking_data in enumerate(bookings):
                try:
                    result = self.predict(booking_data)
                    result["booking_index"] = i
                    results.append(result)
                except Exception as e:
                    # Log error but continue with other predictions
                    logger.error(f"Failed to predict booking {i}: {str(e)}")
                    results.append({
                        "booking_index": i,
                        "error": str(e),
                        "prediction": None,
                        "probability": None,
                        "risk_level": None
                    })
            
            total_time = time.time() - start_time
            avg_time = total_time / len(bookings) if bookings else 0
            
            logger.info(
                f"Batch prediction completed: {len(results)} predictions in "
                f"{total_time:.2f}s (avg: {avg_time*1000:.2f}ms per prediction)"
            )
            
            return results
        
        except Exception as e:
            error_msg = f"Batch prediction failed: {str(e)}"
            logger.error(error_msg)
            raise PredictionError(error_msg)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict containing model metadata and information
        """
        info = {
            "model_loaded": self.model is not None,
            "preprocessor_loaded": self.preprocessor is not None or self.feature_engineer is not None,
            "feature_engineer_loaded": self.feature_engineer is not None,
            "model_path": str(self.model_path),
            "preprocessor_path": str(self.preprocessor_path) if self.preprocessor_path else None,
            "feature_engineer_dir": str(self.feature_engineer_dir) if self.feature_engineer_dir else None,
            **self.metadata
        }
        
        return info
    
    def reload_artifacts(self) -> None:
        """
        Reload model and preprocessor artifacts from disk.
        
        Useful for updating the service with a newly trained model
        without restarting the application.
        
        Raises:
            PredictionError: If reloading fails
        """
        logger.info("Reloading artifacts...")
        self.load_artifacts()
        logger.info("Artifacts reloaded successfully")
