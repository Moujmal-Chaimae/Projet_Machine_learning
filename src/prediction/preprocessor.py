"""
Preprocessor for new booking predictions.

This module provides preprocessing functionality for new booking data,
applying the same transformations that were used during model training.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union, Optional
from pathlib import Path

from src.data_processing.feature_engineer import FeatureEngineer
from src.utils.logger import get_logger
from src.utils.exceptions import DataValidationError, PredictionError

logger = get_logger(__name__)


class Preprocessor:
    """
    Preprocesses new booking data for prediction.
    
    This class loads fitted transformers (encoders, scalers) from training
    and applies the same preprocessing pipeline to new booking data to ensure
    consistency between training and prediction.
    """
    
    def __init__(
        self,
        transformers_path: Optional[str] = None,
        label_encode_cols: Optional[List[str]] = None,
        onehot_encode_cols: Optional[List[str]] = None,
        scale_cols: Optional[List[str]] = None,
        features_to_drop: Optional[List[str]] = None
    ):
        """
        Initialize the Preprocessor.
        
        Args:
            transformers_path: Path to directory containing fitted transformers.
                             If provided, transformers will be loaded automatically.
            label_encode_cols: List of columns to label encode
            onehot_encode_cols: List of columns to one-hot encode
            scale_cols: List of columns to scale
            features_to_drop: List of features to drop before prediction
        
        Raises:
            PredictionError: If initialization fails
        """
        self.feature_engineer = FeatureEngineer()
        self.transformers_loaded = False
        
        # Store configuration
        self.label_encode_cols = label_encode_cols or []
        self.onehot_encode_cols = onehot_encode_cols or []
        self.scale_cols = scale_cols
        self.features_to_drop = features_to_drop or []
        
        logger.info("Preprocessor initialized")
        
        # Load transformers if path provided
        if transformers_path:
            self.load_transformers(transformers_path)
    
    def load_transformers(
        self,
        transformers_path: str,
        prefix: str = "feature_engineer"
    ) -> None:
        """
        Load fitted transformers (encoders, scalers) from disk.
        
        This method loads the transformers that were fitted during training,
        ensuring that new data is preprocessed in exactly the same way.
        
        Args:
            transformers_path: Path to directory containing transformer files
            prefix: Prefix of the transformer files (default: "feature_engineer")
        
        Raises:
            PredictionError: If loading fails
        """
        try:
            transformers_dir = Path(transformers_path)
            
            if not transformers_dir.exists():
                raise FileNotFoundError(
                    f"Transformers directory not found: {transformers_path}"
                )
            
            logger.info(f"Loading transformers from: {transformers_path}")
            
            # Load transformers using FeatureEngineer
            self.feature_engineer.load_transformers(
                input_dir=str(transformers_dir),
                prefix=prefix
            )
            
            self.transformers_loaded = True
            logger.info("Transformers loaded successfully")
            
        except FileNotFoundError as e:
            error_msg = f"Transformer files not found: {str(e)}"
            logger.error(error_msg)
            raise PredictionError(error_msg)
        
        except Exception as e:
            error_msg = f"Failed to load transformers: {str(e)}"
            logger.error(error_msg)
            raise PredictionError(error_msg)
    
    def transform(
        self,
        booking_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]
    ) -> np.ndarray:
        """
        Apply preprocessing transformations to new booking data.
        
        This is the main method that applies the complete preprocessing pipeline:
        1. Convert input to DataFrame
        2. Apply feature engineering (derived features)
        3. Encode categorical variables
        4. Scale numerical features
        5. Drop unnecessary features
        6. Return as numpy array
        
        Args:
            booking_data: Booking data as dict, list of dicts, or DataFrame
        
        Returns:
            np.ndarray: Preprocessed features ready for prediction
        
        Raises:
            PredictionError: If transformation fails
            DataValidationError: If transformers not loaded
        """
        if not self.transformers_loaded:
            raise DataValidationError(
                "Transformers not loaded. Call load_transformers() first."
            )
        
        try:
            logger.debug("Starting preprocessing transformation...")
            
            # Convert input to DataFrame
            df = self._convert_to_dataframe(booking_data)
            
            # Apply feature engineering
            df_transformed = self.apply_feature_engineering(df)
            
            # Encode and scale
            df_transformed = self.encode_and_scale(df_transformed)
            
            # Drop unnecessary features
            df_transformed = self._drop_features(df_transformed)
            
            # Convert to numpy array
            X = df_transformed.values
            
            logger.debug(f"Preprocessing completed. Output shape: {X.shape}")
            
            return X
        
        except Exception as e:
            error_msg = f"Preprocessing transformation failed: {str(e)}"
            logger.error(error_msg)
            raise PredictionError(error_msg)
    
    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features for new bookings.
        
        This method applies the same feature engineering that was done during
        training, creating features like:
        - total_guests
        - total_nights
        - has_children
        - is_long_stay
        - price_per_night_per_guest
        - room_type_match
        - has_special_requests
        
        Args:
            df: Input DataFrame with raw booking data
        
        Returns:
            pd.DataFrame: DataFrame with derived features added
        
        Raises:
            DataValidationError: If required columns are missing
        """
        try:
            logger.debug("Applying feature engineering...")
            
            # Use FeatureEngineer to create derived features
            df_engineered = self.feature_engineer.create_derived_features(df)
            
            # Apply log transformations to skewed features
            df_engineered = self.feature_engineer.apply_transformations(
                df_engineered,
                skewness_threshold=1.0
            )
            
            logger.debug(
                f"Feature engineering completed. Shape: {df_engineered.shape}"
            )
            
            return df_engineered
        
        except Exception as e:
            error_msg = f"Feature engineering failed: {str(e)}"
            logger.error(error_msg)
            raise DataValidationError(error_msg)
    
    def encode_and_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply encoding and scaling transformations.
        
        This method applies the fitted encoders and scalers to transform
        categorical and numerical features in the same way as training data.
        
        Args:
            df: Input DataFrame with engineered features
        
        Returns:
            pd.DataFrame: DataFrame with encoded and scaled features
        
        Raises:
            DataValidationError: If encoding or scaling fails
        """
        try:
            logger.debug("Applying encoding and scaling...")
            
            # Encode categorical variables
            df_encoded = self.feature_engineer.encode_categorical(
                df,
                label_encode_cols=self.label_encode_cols,
                onehot_encode_cols=self.onehot_encode_cols,
                fit=False  # Use fitted encoders
            )
            
            # Scale numerical features
            df_scaled = self.feature_engineer.scale_numerical(
                df_encoded,
                columns=self.scale_cols,
                fit=False  # Use fitted scaler
            )
            
            logger.debug(
                f"Encoding and scaling completed. Shape: {df_scaled.shape}"
            )
            
            return df_scaled
        
        except Exception as e:
            error_msg = f"Encoding and scaling failed: {str(e)}"
            logger.error(error_msg)
            raise DataValidationError(error_msg)
    
    def _convert_to_dataframe(
        self,
        booking_data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Convert input booking data to DataFrame format.
        
        Args:
            booking_data: Booking data in various formats
        
        Returns:
            pd.DataFrame: Booking data as DataFrame
        
        Raises:
            DataValidationError: If conversion fails
        """
        try:
            if isinstance(booking_data, pd.DataFrame):
                return booking_data.copy()
            
            elif isinstance(booking_data, dict):
                # Single booking as dict
                return pd.DataFrame([booking_data])
            
            elif isinstance(booking_data, list):
                # Multiple bookings as list of dicts
                if not booking_data:
                    raise DataValidationError("Empty booking list provided")
                
                if not all(isinstance(item, dict) for item in booking_data):
                    raise DataValidationError(
                        "All items in booking list must be dictionaries"
                    )
                
                return pd.DataFrame(booking_data)
            
            else:
                raise DataValidationError(
                    f"Unsupported booking data type: {type(booking_data)}"
                )
        
        except Exception as e:
            error_msg = f"Failed to convert booking data to DataFrame: {str(e)}"
            logger.error(error_msg)
            raise DataValidationError(error_msg)
    
    def _drop_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop features that should not be used for prediction.
        
        This includes features like:
        - Target variable (is_canceled)
        - Features that cause data leakage
        - Features specified in configuration
        
        Args:
            df: Input DataFrame
        
        Returns:
            pd.DataFrame: DataFrame with features dropped
        """
        df_dropped = df.copy()
        
        # Always drop target variable if present
        if 'is_canceled' in df_dropped.columns:
            df_dropped = df_dropped.drop(columns=['is_canceled'])
            logger.debug("Dropped target variable 'is_canceled'")
        
        # Drop configured features
        if self.features_to_drop:
            existing_features = [
                f for f in self.features_to_drop if f in df_dropped.columns
            ]
            
            if existing_features:
                df_dropped = df_dropped.drop(columns=existing_features)
                logger.debug(f"Dropped {len(existing_features)} configured features")
        
        return df_dropped
    
    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names after preprocessing.
        
        Returns:
            List[str]: List of feature names
        
        Raises:
            DataValidationError: If transformers not loaded
        """
        if not self.transformers_loaded:
            raise DataValidationError(
                "Transformers not loaded. Call load_transformers() first."
            )
        
        return self.feature_engineer.get_feature_names()
    
    def get_preprocessor_info(self) -> Dict[str, Any]:
        """
        Get information about the preprocessor configuration.
        
        Returns:
            Dict containing preprocessor information
        """
        info = {
            "transformers_loaded": self.transformers_loaded,
            "label_encode_cols": self.label_encode_cols,
            "onehot_encode_cols": self.onehot_encode_cols,
            "scale_cols": self.scale_cols,
            "features_to_drop": self.features_to_drop
        }
        
        if self.transformers_loaded:
            info.update(self.feature_engineer.get_feature_info())
        
        return info
    
    def validate_input_features(self, df: pd.DataFrame) -> bool:
        """
        Validate that input DataFrame has required features.
        
        Args:
            df: Input DataFrame to validate
        
        Returns:
            bool: True if validation passes
        
        Raises:
            DataValidationError: If validation fails
        """
        # Check for basic required columns for feature engineering
        required_cols = [
            'adults', 'children', 'babies',
            'stays_in_weekend_nights', 'stays_in_week_nights',
            'adr'
        ]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise DataValidationError(
                f"Missing required columns: {missing_cols}"
            )
        
        return True
