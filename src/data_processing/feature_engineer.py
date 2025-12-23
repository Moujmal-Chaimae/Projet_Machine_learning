"""
Feature engineering module for the hotel cancellation prediction system.
Handles feature creation, encoding, scaling, and transformations.
"""

import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.utils.logger import get_logger
from src.utils.exceptions import DataValidationError

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Creates and transforms features for hotel booking data.
    
    This class provides methods to:
    - Create derived features (total_guests, total_nights, etc.)
    - Encode categorical variables (label encoding and one-hot encoding)
    - Scale numerical features using StandardScaler
    - Apply transformations (log transformation for skewed features)
    - Save and load fitted transformers for prediction
    """
    
    def __init__(self):
        """Initialize the FeatureEngineer."""
        self.label_encoders = {}
        self.onehot_encoder = None
        self.scaler = None
        self.feature_names = []
        self.onehot_feature_names = []
        self.fitted = False
        self.log_transformed_cols = []  # Track which columns were log-transformed during fit
        self.label_encode_cols = []  # Track which columns were label-encoded during fit
        self.onehot_encode_cols = []  # Track which columns were one-hot-encoded during fit
        
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from existing columns.
        
        Creates the following features:
        - total_guests: adults + children + babies
        - total_nights: stays_in_weekend_nights + stays_in_week_nights
        - has_children: 1 if children or babies > 0, else 0
        - is_long_stay: 1 if total_nights > 7, else 0
        - price_per_night_per_guest: adr / total_guests (if total_guests > 0)
        - room_type_match: 1 if reserved_room_type == assigned_room_type, else 0
        - has_special_requests: 1 if total_of_special_requests > 0, else 0
        
        Args:
            df: Input DataFrame
        
        Returns:
            pd.DataFrame: DataFrame with derived features added
        
        Raises:
            DataValidationError: If required columns are missing
        """
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise DataValidationError("Input DataFrame is empty")
        
        logger.info("Creating derived features...")
        
        df_features = df.copy()
        
        # Check for required columns
        required_cols = ['adults', 'children', 'babies', 'stays_in_weekend_nights', 
                        'stays_in_week_nights', 'adr']
        missing_cols = [col for col in required_cols if col not in df_features.columns]
        
        if missing_cols:
            raise DataValidationError(
                f"Missing required columns for feature engineering: {missing_cols}"
            )
        
        # 1. Total guests
        df_features['total_guests'] = (
            df_features['adults'].fillna(0) + 
            df_features['children'].fillna(0) + 
            df_features['babies'].fillna(0)
        )
        logger.debug(f"Created 'total_guests' feature (range: {df_features['total_guests'].min()}-{df_features['total_guests'].max()})")
        
        # 2. Total nights
        df_features['total_nights'] = (
            df_features['stays_in_weekend_nights'].fillna(0) + 
            df_features['stays_in_week_nights'].fillna(0)
        )
        logger.debug(f"Created 'total_nights' feature (range: {df_features['total_nights'].min()}-{df_features['total_nights'].max()})")
        
        # 3. Has children
        df_features['has_children'] = (
            ((df_features['children'].fillna(0) > 0) | 
             (df_features['babies'].fillna(0) > 0))
        ).astype(int)
        logger.debug(f"Created 'has_children' feature ({df_features['has_children'].sum()} bookings with children)")
        
        # 4. Is long stay
        df_features['is_long_stay'] = (df_features['total_nights'] > 7).astype(int)
        logger.debug(f"Created 'is_long_stay' feature ({df_features['is_long_stay'].sum()} long stays)")
        
        # 5. Price per night per guest
        # Avoid division by zero
        df_features['price_per_night_per_guest'] = 0.0
        valid_mask = (df_features['total_guests'] > 0) & (df_features['total_nights'] > 0)
        df_features.loc[valid_mask, 'price_per_night_per_guest'] = (
            df_features.loc[valid_mask, 'adr'] / 
            (df_features.loc[valid_mask, 'total_guests'] * df_features.loc[valid_mask, 'total_nights'])
        )
        logger.debug(f"Created 'price_per_night_per_guest' feature")
        
        # 6. Room type match
        if 'reserved_room_type' in df_features.columns and 'assigned_room_type' in df_features.columns:
            df_features['room_type_match'] = (
                df_features['reserved_room_type'] == df_features['assigned_room_type']
            ).astype(int)
            logger.debug(f"Created 'room_type_match' feature ({df_features['room_type_match'].sum()} matches)")
        
        # 7. Has special requests
        if 'total_of_special_requests' in df_features.columns:
            df_features['has_special_requests'] = (
                df_features['total_of_special_requests'] > 0
            ).astype(int)
            logger.debug(f"Created 'has_special_requests' feature ({df_features['has_special_requests'].sum()} with requests)")
        
        logger.info(f"Created {7} derived features. New shape: {df_features.shape}")
        
        return df_features
    
    def encode_categorical(
        self, 
        df: pd.DataFrame, 
        label_encode_cols: Optional[List[str]] = None,
        onehot_encode_cols: Optional[List[str]] = None,
        fit: bool = True,
        encode_all_categorical: bool = True
    ) -> pd.DataFrame:
        """
        Encode categorical variables using label encoding or one-hot encoding.
        
        Args:
            df: Input DataFrame
            label_encode_cols: List of columns to label encode
            onehot_encode_cols: List of columns to one-hot encode
            fit: If True, fit the encoders. If False, use existing fitted encoders.
        
        Returns:
            pd.DataFrame: DataFrame with encoded categorical variables
        
        Raises:
            DataValidationError: If columns are missing or encoders not fitted
        """
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise DataValidationError("Input DataFrame is empty")
        
        df_encoded = df.copy()
        
        # Label encoding
        if label_encode_cols:
            logger.info(f"Label encoding {len(label_encode_cols)} columns: {label_encode_cols}")
            
            for col in label_encode_cols:
                if col not in df_encoded.columns:
                    logger.warning(f"Column '{col}' not found in DataFrame, skipping")
                    continue
                
                if fit:
                    # Fit new encoder
                    le = LabelEncoder()
                    # Handle missing values by filling with a placeholder
                    df_encoded[col] = df_encoded[col].fillna('Unknown')
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.label_encoders[col] = le
                    logger.debug(f"  {col}: {len(le.classes_)} unique values encoded")
                else:
                    # Use existing encoder
                    if col not in self.label_encoders:
                        raise DataValidationError(
                            f"Encoder for column '{col}' not fitted. Call with fit=True first."
                        )
                    le = self.label_encoders[col]
                    df_encoded[col] = df_encoded[col].fillna('Unknown')
                    # Handle unseen categories
                    df_encoded[col] = df_encoded[col].apply(
                        lambda x: x if x in le.classes_ else 'Unknown'
                    )
                    df_encoded[col] = le.transform(df_encoded[col].astype(str))
        
        # One-hot encoding
        if onehot_encode_cols:
            logger.info(f"One-hot encoding {len(onehot_encode_cols)} columns: {onehot_encode_cols}")
            
            # Filter columns that exist
            existing_onehot_cols = [col for col in onehot_encode_cols if col in df_encoded.columns]
            missing_onehot_cols = set(onehot_encode_cols) - set(existing_onehot_cols)
            
            if missing_onehot_cols:
                logger.warning(f"Columns not found: {missing_onehot_cols}")
            
            if existing_onehot_cols:
                if fit:
                    # Fit new encoder
                    self.onehot_encoder = OneHotEncoder(
                        sparse_output=False, 
                        handle_unknown='ignore',
                        drop='first'  # Drop first category to avoid multicollinearity
                    )
                    
                    # Fill missing values
                    for col in existing_onehot_cols:
                        df_encoded[col] = df_encoded[col].fillna('Unknown')
                    
                    # Fit and transform
                    onehot_encoded = self.onehot_encoder.fit_transform(
                        df_encoded[existing_onehot_cols]
                    )
                    
                    # Generate feature names
                    self.onehot_feature_names = []
                    for i, col in enumerate(existing_onehot_cols):
                        categories = self.onehot_encoder.categories_[i][1:]  # Skip first (dropped)
                        for cat in categories:
                            self.onehot_feature_names.append(f"{col}_{cat}")
                    
                    # Create DataFrame with one-hot encoded features
                    onehot_df = pd.DataFrame(
                        onehot_encoded,
                        columns=self.onehot_feature_names,
                        index=df_encoded.index
                    )
                    
                    # Drop original columns and concatenate one-hot encoded columns
                    df_encoded = df_encoded.drop(columns=existing_onehot_cols)
                    df_encoded = pd.concat([df_encoded, onehot_df], axis=1)
                    
                    logger.debug(f"  Created {len(self.onehot_feature_names)} one-hot encoded features")
                
                else:
                    # Use existing encoder
                    if self.onehot_encoder is None:
                        raise DataValidationError(
                            "One-hot encoder not fitted. Call with fit=True first."
                        )
                    
                    # Fill missing values
                    for col in existing_onehot_cols:
                        df_encoded[col] = df_encoded[col].fillna('Unknown')
                    
                    # Transform
                    onehot_encoded = self.onehot_encoder.transform(
                        df_encoded[existing_onehot_cols]
                    )
                    
                    # Create DataFrame with one-hot encoded features
                    onehot_df = pd.DataFrame(
                        onehot_encoded,
                        columns=self.onehot_feature_names,
                        index=df_encoded.index
                    )
                    
                    # Drop original columns and concatenate one-hot encoded columns
                    df_encoded = df_encoded.drop(columns=existing_onehot_cols)
                    df_encoded = pd.concat([df_encoded, onehot_df], axis=1)
        
        # Auto-encode remaining categorical columns
        if encode_all_categorical:
            # Find remaining object/categorical columns
            remaining_cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if remaining_cat_cols:
                logger.info(f"Auto-encoding {len(remaining_cat_cols)} remaining categorical columns: {remaining_cat_cols}")
                
                for col in remaining_cat_cols:
                    if fit:
                        # Fit new encoder
                        le = LabelEncoder()
                        df_encoded[col] = df_encoded[col].fillna('Unknown')
                        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                        self.label_encoders[col] = le
                        logger.debug(f"  Auto-encoded {col}: {len(le.classes_)} unique values")
                    else:
                        # Use existing encoder
                        if col in self.label_encoders:
                            le = self.label_encoders[col]
                            df_encoded[col] = df_encoded[col].fillna('Unknown')
                            # Handle unseen categories
                            df_encoded[col] = df_encoded[col].apply(
                                lambda x: x if x in le.classes_ else 'Unknown'
                            )
                            df_encoded[col] = le.transform(df_encoded[col].astype(str))
                        else:
                            logger.warning(f"No encoder found for {col}, creating new one")
                            le = LabelEncoder()
                            df_encoded[col] = df_encoded[col].fillna('Unknown')
                            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                            self.label_encoders[col] = le
        
        logger.info(f"Encoding completed. New shape: {df_encoded.shape}")
        
        return df_encoded
    
    def scale_numerical(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler (z-score normalization).
        
        Args:
            df: Input DataFrame
            columns: List of numerical columns to scale. If None, auto-detect numerical columns.
            fit: If True, fit the scaler. If False, use existing fitted scaler.
        
        Returns:
            pd.DataFrame: DataFrame with scaled numerical features
        
        Raises:
            DataValidationError: If scaler not fitted when fit=False
        """
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise DataValidationError("Input DataFrame is empty")
        
        df_scaled = df.copy()
        
        # Auto-detect numerical columns if not specified
        if columns is None:
            columns = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude binary features and target variable
            exclude_cols = ['is_canceled', 'has_children', 'is_long_stay', 
                           'room_type_match', 'has_special_requests', 'is_repeated_guest']
            columns = [col for col in columns if col not in exclude_cols]
            logger.info(f"Auto-detected {len(columns)} numerical columns for scaling")
        
        if not columns:
            logger.warning("No numerical columns to scale")
            return df_scaled
        
        # Filter columns that exist
        existing_cols = [col for col in columns if col in df_scaled.columns]
        missing_cols = set(columns) - set(existing_cols)
        
        if missing_cols:
            logger.warning(f"Columns not found for scaling: {missing_cols}")
        
        if not existing_cols:
            logger.warning("No valid columns to scale")
            return df_scaled
        
        logger.info(f"Scaling {len(existing_cols)} numerical columns")
        
        if fit:
            # Fit new scaler
            self.scaler = StandardScaler()
            df_scaled[existing_cols] = self.scaler.fit_transform(df_scaled[existing_cols])
            logger.debug(f"  Fitted scaler with mean: {self.scaler.mean_[:5]}... (showing first 5)")
            logger.debug(f"  Scale: {self.scaler.scale_[:5]}... (showing first 5)")
        else:
            # Use existing scaler
            if self.scaler is None:
                raise DataValidationError(
                    "Scaler not fitted. Call with fit=True first."
                )
            df_scaled[existing_cols] = self.scaler.transform(df_scaled[existing_cols])
        
        logger.info(f"Scaling completed for {len(existing_cols)} columns")
        
        return df_scaled
    
    def apply_transformations(
        self, 
        df: pd.DataFrame, 
        skewness_threshold: float = 1.0,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Apply log transformation to skewed numerical features.
        
        Features with skewness > threshold will be log-transformed.
        Only positive-valued features are transformed.
        
        Args:
            df: Input DataFrame
            skewness_threshold: Threshold for applying log transformation (default: 1.0)
            fit: If True, determine which columns to transform. If False, use stored columns.
        
        Returns:
            pd.DataFrame: DataFrame with transformed features
        """
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise DataValidationError("Input DataFrame is empty")
        
        df_transformed = df.copy()
        
        if fit:
            # Determine which columns to transform
            self.log_transformed_cols = []
            
            # Get numerical columns
            numerical_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
            
            # Exclude binary and target columns
            exclude_cols = ['is_canceled', 'has_children', 'is_long_stay', 
                           'room_type_match', 'has_special_requests', 'is_repeated_guest']
            numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
            
            logger.info(f"Checking skewness for {len(numerical_cols)} numerical columns...")
            
            transformed_count = 0
            
            for col in numerical_cols:
                if col not in df_transformed.columns:
                    continue
                
                # Calculate skewness
                skewness = df_transformed[col].skew()
                
                # Check if feature has positive values only (required for log transform)
                min_value = df_transformed[col].min()
                
                if abs(skewness) > skewness_threshold and min_value > 0:
                    # Apply log transformation
                    df_transformed[f'{col}_log'] = np.log1p(df_transformed[col])
                    self.log_transformed_cols.append(col)
                    logger.debug(f"  {col}: skewness={skewness:.2f} -> applied log transformation")
                    transformed_count += 1
                elif abs(skewness) > skewness_threshold and min_value <= 0:
                    logger.debug(f"  {col}: skewness={skewness:.2f} but has non-positive values, skipping")
            
            logger.info(f"Applied log transformation to {transformed_count} features")
        else:
            # Use stored columns from fit
            logger.info(f"Applying log transformation to {len(self.log_transformed_cols)} pre-determined columns...")
            
            for col in self.log_transformed_cols:
                if col in df_transformed.columns:
                    df_transformed[f'{col}_log'] = np.log1p(df_transformed[col])
                else:
                    logger.warning(f"Column '{col}' not found for log transformation")
        
        return df_transformed
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        label_encode_cols: Optional[List[str]] = None,
        onehot_encode_cols: Optional[List[str]] = None,
        scale_cols: Optional[List[str]] = None,
        apply_log_transform: bool = True,
        skewness_threshold: float = 1.0
    ) -> pd.DataFrame:
        """
        Fit all transformers and transform the data in one step.
        
        This method:
        1. Creates derived features
        2. Applies log transformations to skewed features
        3. Encodes categorical variables
        4. Scales numerical features
        
        Args:
            df: Input DataFrame
            label_encode_cols: Columns to label encode
            onehot_encode_cols: Columns to one-hot encode
            scale_cols: Columns to scale (if None, auto-detect)
            apply_log_transform: Whether to apply log transformation
            skewness_threshold: Threshold for log transformation
        
        Returns:
            pd.DataFrame: Fully transformed DataFrame
        """
        logger.info("Starting fit_transform pipeline...")
        
        # Store encoding configuration for later use in transform
        self.label_encode_cols = label_encode_cols if label_encode_cols else []
        self.onehot_encode_cols = onehot_encode_cols if onehot_encode_cols else []
        
        # 1. Create derived features
        df_transformed = self.create_derived_features(df)
        
        # 2. Apply log transformations
        if apply_log_transform:
            df_transformed = self.apply_transformations(df_transformed, skewness_threshold, fit=True)
        
        # 3. Encode categorical variables
        df_transformed = self.encode_categorical(
            df_transformed,
            label_encode_cols=label_encode_cols,
            onehot_encode_cols=onehot_encode_cols,
            fit=True
        )
        
        # 4. Scale numerical features
        df_transformed = self.scale_numerical(df_transformed, columns=scale_cols, fit=True)
        
        # Store feature names
        self.feature_names = df_transformed.columns.tolist()
        self.fitted = True
        
        logger.info(f"fit_transform completed. Final shape: {df_transformed.shape}")
        logger.info(f"Total features: {len(self.feature_names)}")
        
        return df_transformed
    
    def transform(
        self,
        df: pd.DataFrame,
        label_encode_cols: Optional[List[str]] = None,
        onehot_encode_cols: Optional[List[str]] = None,
        scale_cols: Optional[List[str]] = None,
        apply_log_transform: bool = True,
        skewness_threshold: float = 1.0
    ) -> pd.DataFrame:
        """
        Transform new data using fitted transformers.
        
        This method applies the same transformations as fit_transform but uses
        already fitted encoders and scalers.
        
        Args:
            df: Input DataFrame
            label_encode_cols: Columns to label encode (if None, use saved from fit)
            onehot_encode_cols: Columns to one-hot encode (if None, use saved from fit)
            scale_cols: Columns to scale
            apply_log_transform: Whether to apply log transformation
            skewness_threshold: Threshold for log transformation
        
        Returns:
            pd.DataFrame: Transformed DataFrame
        
        Raises:
            DataValidationError: If transformers not fitted
        """
        if not self.fitted:
            raise DataValidationError(
                "Transformers not fitted. Call fit_transform() first."
            )
        
        logger.info("Starting transform pipeline...")
        
        # Use saved encoding configuration if not provided
        if label_encode_cols is None:
            label_encode_cols = self.label_encode_cols
        if onehot_encode_cols is None:
            onehot_encode_cols = self.onehot_encode_cols
        
        # 1. Create derived features
        df_transformed = self.create_derived_features(df)
        
        # 2. Apply log transformations (use same columns as during fit)
        if apply_log_transform:
            df_transformed = self.apply_transformations(df_transformed, skewness_threshold, fit=False)
        
        # 3. Encode categorical variables
        df_transformed = self.encode_categorical(
            df_transformed,
            label_encode_cols=label_encode_cols,
            onehot_encode_cols=onehot_encode_cols,
            fit=False
        )
        
        # 3.5. Ensure columns match those from training BEFORE scaling
        if self.feature_names:
            # Add missing columns with zeros
            for col in self.feature_names:
                if col not in df_transformed.columns:
                    df_transformed[col] = 0
                    logger.debug(f"Added missing column '{col}' with default value 0")
            
            # Remove extra columns not in training
            extra_cols = set(df_transformed.columns) - set(self.feature_names)
            if extra_cols:
                logger.debug(f"Removing extra columns not in training: {extra_cols}")
                df_transformed = df_transformed.drop(columns=list(extra_cols))
            
            # Reorder columns to match training
            df_transformed = df_transformed[self.feature_names]
            logger.debug(f"Reordered columns to match training feature order ({len(self.feature_names)} features)")
        
        # 4. Scale numerical features (now columns are in correct order)
        df_transformed = self.scale_numerical(df_transformed, columns=scale_cols, fit=False)
        
        logger.info(f"transform completed. Final shape: {df_transformed.shape}")
        
        return df_transformed
    
    def save_transformers(self, output_dir: str, prefix: str = "feature_engineer") -> None:
        """
        Save fitted encoders and scalers to disk for later use in prediction.
        
        Args:
            output_dir: Directory to save the transformers
            prefix: Prefix for the saved files
        
        Raises:
            DataValidationError: If transformers not fitted
        """
        if not self.fitted:
            raise DataValidationError(
                "Transformers not fitted. Call fit_transform() first."
            )
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save label encoders
        if self.label_encoders:
            label_encoder_path = os.path.join(output_dir, f"{prefix}_label_encoders.pkl")
            joblib.dump(self.label_encoders, label_encoder_path)
            logger.info(f"Saved label encoders to {label_encoder_path}")
        
        # Save one-hot encoder
        if self.onehot_encoder is not None:
            onehot_encoder_path = os.path.join(output_dir, f"{prefix}_onehot_encoder.pkl")
            joblib.dump(self.onehot_encoder, onehot_encoder_path)
            joblib.dump(self.onehot_feature_names, 
                       os.path.join(output_dir, f"{prefix}_onehot_feature_names.pkl"))
            logger.info(f"Saved one-hot encoder to {onehot_encoder_path}")
        
        # Save scaler
        if self.scaler is not None:
            scaler_path = os.path.join(output_dir, f"{prefix}_scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")
        
        # Save feature names
        feature_names_path = os.path.join(output_dir, f"{prefix}_feature_names.pkl")
        joblib.dump(self.feature_names, feature_names_path)
        logger.info(f"Saved feature names to {feature_names_path}")
        
        # Save log-transformed columns
        log_transformed_path = os.path.join(output_dir, f"{prefix}_log_transformed_cols.pkl")
        joblib.dump(self.log_transformed_cols, log_transformed_path)
        logger.info(f"Saved log-transformed columns to {log_transformed_path}")
        
        # Save encoding configuration
        encoding_config_path = os.path.join(output_dir, f"{prefix}_encoding_config.pkl")
        encoding_config = {
            'label_encode_cols': self.label_encode_cols,
            'onehot_encode_cols': self.onehot_encode_cols
        }
        joblib.dump(encoding_config, encoding_config_path)
        logger.info(f"Saved encoding configuration to {encoding_config_path}")
        
        logger.info(f"All transformers saved to {output_dir}")
    
    def load_transformers(self, input_dir: str, prefix: str = "feature_engineer") -> None:
        """
        Load fitted encoders and scalers from disk.
        
        Args:
            input_dir: Directory containing the saved transformers
            prefix: Prefix of the saved files
        
        Raises:
            FileNotFoundError: If transformer files not found
        """
        logger.info(f"Loading transformers from {input_dir}...")
        
        # Load label encoders
        label_encoder_path = os.path.join(input_dir, f"{prefix}_label_encoders.pkl")
        if os.path.exists(label_encoder_path):
            self.label_encoders = joblib.load(label_encoder_path)
            logger.info(f"Loaded {len(self.label_encoders)} label encoders")
        
        # Load one-hot encoder
        onehot_encoder_path = os.path.join(input_dir, f"{prefix}_onehot_encoder.pkl")
        if os.path.exists(onehot_encoder_path):
            self.onehot_encoder = joblib.load(onehot_encoder_path)
            self.onehot_feature_names = joblib.load(
                os.path.join(input_dir, f"{prefix}_onehot_feature_names.pkl")
            )
            logger.info(f"Loaded one-hot encoder with {len(self.onehot_feature_names)} features")
        
        # Load scaler
        scaler_path = os.path.join(input_dir, f"{prefix}_scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info("Loaded scaler")
        
        # Load feature names
        feature_names_path = os.path.join(input_dir, f"{prefix}_feature_names.pkl")
        if os.path.exists(feature_names_path):
            self.feature_names = joblib.load(feature_names_path)
            # Remove target column if present (it should not be in features)
            if 'is_canceled' in self.feature_names:
                self.feature_names.remove('is_canceled')
                logger.info("Removed target column 'is_canceled' from feature names")
            logger.info(f"Loaded {len(self.feature_names)} feature names")
        
        # Load log-transformed columns
        log_transformed_path = os.path.join(input_dir, f"{prefix}_log_transformed_cols.pkl")
        if os.path.exists(log_transformed_path):
            self.log_transformed_cols = joblib.load(log_transformed_path)
            logger.info(f"Loaded {len(self.log_transformed_cols)} log-transformed columns")
        else:
            self.log_transformed_cols = []
        
        # Load encoding configuration
        encoding_config_path = os.path.join(input_dir, f"{prefix}_encoding_config.pkl")
        if os.path.exists(encoding_config_path):
            encoding_config = joblib.load(encoding_config_path)
            self.label_encode_cols = encoding_config.get('label_encode_cols', [])
            self.onehot_encode_cols = encoding_config.get('onehot_encode_cols', [])
            logger.info(f"Loaded encoding configuration: {len(self.label_encode_cols)} label-encoded, {len(self.onehot_encode_cols)} one-hot-encoded columns")
        else:
            logger.warning("Encoding configuration not found, using empty lists")
            self.label_encode_cols = []
            self.onehot_encode_cols = []
        
        self.fitted = True
        logger.info("All transformers loaded successfully")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names after transformation.
        
        Returns:
            List[str]: List of feature names
        """
        return self.feature_names.copy()
    
    def get_feature_info(self) -> Dict:
        """
        Get information about the fitted transformers.
        
        Returns:
            dict: Dictionary containing transformer information
        """
        info = {
            'fitted': self.fitted,
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'label_encoded_columns': list(self.label_encoders.keys()),
            'onehot_encoded_features': self.onehot_feature_names,
            'scaler_fitted': self.scaler is not None
        }
        
        if self.scaler is not None:
            info['scaler_mean'] = self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None
            info['scaler_scale'] = self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None
        
        return info
