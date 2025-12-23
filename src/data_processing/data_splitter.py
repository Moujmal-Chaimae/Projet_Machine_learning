"""
Data splitting module for the hotel cancellation prediction system.
Handles splitting data into training and testing sets with stratification.
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from src.utils.logger import get_logger
from src.utils.exceptions import DataValidationError

logger = get_logger(__name__)


class DataSplitter:
    """
    Splits hotel booking data into training and testing sets.
    
    This class provides methods to:
    - Split data with stratification on target variable
    - Save processed datasets to disk using pickle
    - Log dataset shapes and class distributions
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the DataSplitter.
        
        Args:
            test_size: Proportion of dataset to include in test split (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.test_size = test_size
        self.random_state = random_state
        logger.info(f"DataSplitter initialized with test_size={test_size}, random_state={random_state}")
    
    def split_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'is_canceled',
        stratify: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets with optional stratification.
        
        Args:
            df: Input DataFrame containing features and target
            target_column: Name of the target variable column (default: 'is_canceled')
            stratify: Whether to stratify split on target variable (default: True)
        
        Returns:
            Tuple containing:
                - X_train: Training features
                - X_test: Testing features
                - y_train: Training labels
                - y_test: Testing labels
        
        Raises:
            DataValidationError: If input is invalid or target column is missing
        """
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise DataValidationError("Input DataFrame is empty")
        
        if target_column not in df.columns:
            raise DataValidationError(
                f"Target column '{target_column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
        
        logger.info(f"Starting data split with test_size={self.test_size}")
        logger.info(f"Input dataset shape: {df.shape}")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        # Check for missing values in target
        if y.isnull().any():
            missing_count = y.isnull().sum()
            logger.warning(
                f"Target variable has {missing_count} missing values. "
                f"These rows will be removed."
            )
            # Remove rows with missing target
            valid_indices = ~y.isnull()
            X = X[valid_indices]
            y = y[valid_indices]
            logger.info(f"After removing missing targets - X shape: {X.shape}, y shape: {y.shape}")
        
        # Log class distribution before split
        class_counts = y.value_counts().sort_index()
        class_distribution = y.value_counts(normalize=True).sort_index()
        
        logger.info("Class distribution in full dataset:")
        for class_label, count in class_counts.items():
            percentage = class_distribution[class_label] * 100
            logger.info(f"  Class {class_label}: {count} samples ({percentage:.2f}%)")
        
        # Check if stratification is possible
        stratify_param = None
        if stratify:
            # Check if we have enough samples in each class for stratification
            min_class_count = class_counts.min()
            min_samples_needed = int(1 / self.test_size) + 1
            
            if min_class_count < min_samples_needed:
                logger.warning(
                    f"Insufficient samples in minority class ({min_class_count}) "
                    f"for stratification with test_size={self.test_size}. "
                    f"Proceeding without stratification."
                )
                stratify = False
            else:
                stratify_param = y
                logger.info("Stratification enabled on target variable")
        else:
            logger.info("Stratification disabled")
        
        # Perform train-test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify_param
            )
            
            logger.info("Data split completed successfully")
            
        except ValueError as e:
            logger.error(f"Error during train-test split: {e}")
            raise DataValidationError(f"Failed to split data: {e}")
        
        # Log shapes of resulting datasets
        logger.info(f"Training set - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"Testing set - X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        # Log class distributions in train and test sets
        self._log_class_distribution(y_train, y_test)
        
        return X_train, X_test, y_train, y_test
    
    def _log_class_distribution(self, y_train: pd.Series, y_test: pd.Series) -> None:
        """
        Log class distributions for training and testing sets.
        
        Args:
            y_train: Training labels
            y_test: Testing labels
        """
        # Training set distribution
        train_counts = y_train.value_counts().sort_index()
        train_distribution = y_train.value_counts(normalize=True).sort_index()
        
        logger.info("Class distribution in training set:")
        for class_label, count in train_counts.items():
            percentage = train_distribution[class_label] * 100
            logger.info(f"  Class {class_label}: {count} samples ({percentage:.2f}%)")
        
        # Testing set distribution
        test_counts = y_test.value_counts().sort_index()
        test_distribution = y_test.value_counts(normalize=True).sort_index()
        
        logger.info("Class distribution in testing set:")
        for class_label, count in test_counts.items():
            percentage = test_distribution[class_label] * 100
            logger.info(f"  Class {class_label}: {count} samples ({percentage:.2f}%)")
        
        # Check if distributions are similar (for stratification verification)
        if len(train_distribution) == len(test_distribution):
            max_diff = max(abs(train_distribution - test_distribution))
            logger.info(f"Maximum distribution difference between train and test: {max_diff:.4f}")
            
            if max_diff > 0.05:
                logger.warning(
                    f"Class distributions differ by more than 5% between train and test sets. "
                    f"Consider using stratification."
                )
    
    def save_splits(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        output_dir: str = "data/processed/"
    ) -> None:
        """
        Save processed datasets to disk using pickle.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labels
            output_dir: Directory to save the datasets (default: 'data/processed/')
        
        Raises:
            IOError: If unable to save files
        """
        logger.info(f"Saving processed datasets to {output_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define file paths
        files_to_save = {
            'X_train.pkl': X_train,
            'X_test.pkl': X_test,
            'y_train.pkl': y_train,
            'y_test.pkl': y_test
        }
        
        # Save each dataset
        for filename, data in files_to_save.items():
            filepath = os.path.join(output_dir, filename)
            
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
                
                file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                logger.info(f"  Saved {filename} ({file_size_mb:.2f} MB)")
                
            except Exception as e:
                error_msg = f"Failed to save {filename}: {e}"
                logger.error(error_msg)
                raise IOError(error_msg)
        
        logger.info(f"All datasets saved successfully to {output_dir}")
        
        # Log summary
        self._log_save_summary(X_train, X_test, y_train, y_test, output_dir)
    
    def _log_save_summary(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        output_dir: str
    ) -> None:
        """
        Log a summary of saved datasets.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labels
            output_dir: Directory where datasets were saved
        """
        logger.info("=" * 60)
        logger.info("DATASET SPLIT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Test size: {self.test_size} ({self.test_size * 100:.0f}%)")
        logger.info(f"Random state: {self.random_state}")
        logger.info("")
        logger.info("Dataset shapes:")
        logger.info(f"  X_train: {X_train.shape}")
        logger.info(f"  X_test: {X_test.shape}")
        logger.info(f"  y_train: {y_train.shape}")
        logger.info(f"  y_test: {y_test.shape}")
        logger.info("")
        logger.info(f"Total samples: {len(X_train) + len(X_test)}")
        logger.info(f"Training samples: {len(X_train)} ({len(X_train)/(len(X_train)+len(X_test))*100:.1f}%)")
        logger.info(f"Testing samples: {len(X_test)} ({len(X_test)/(len(X_train)+len(X_test))*100:.1f}%)")
        logger.info(f"Number of features: {X_train.shape[1]}")
        logger.info("=" * 60)
    
    def load_splits(
        self,
        input_dir: str = "data/processed/"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load processed datasets from disk.
        
        Args:
            input_dir: Directory containing the saved datasets (default: 'data/processed/')
        
        Returns:
            Tuple containing:
                - X_train: Training features
                - X_test: Testing features
                - y_train: Training labels
                - y_test: Testing labels
        
        Raises:
            FileNotFoundError: If dataset files are not found
            IOError: If unable to load files
        """
        logger.info(f"Loading processed datasets from {input_dir}")
        
        # Define file paths
        files_to_load = ['X_train.pkl', 'X_test.pkl', 'y_train.pkl', 'y_test.pkl']
        
        # Check if all files exist
        missing_files = []
        for filename in files_to_load:
            filepath = os.path.join(input_dir, filename)
            if not os.path.exists(filepath):
                missing_files.append(filename)
        
        if missing_files:
            error_msg = f"Missing dataset files: {missing_files}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Load datasets
        datasets = {}
        for filename in files_to_load:
            filepath = os.path.join(input_dir, filename)
            
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                datasets[filename.replace('.pkl', '')] = data
                logger.info(f"  Loaded {filename}")
                
            except Exception as e:
                error_msg = f"Failed to load {filename}: {e}"
                logger.error(error_msg)
                raise IOError(error_msg)
        
        X_train = datasets['X_train']
        X_test = datasets['X_test']
        y_train = datasets['y_train']
        y_test = datasets['y_test']
        
        logger.info("All datasets loaded successfully")
        logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def split_and_save(
        self,
        df: pd.DataFrame,
        target_column: str = 'is_canceled',
        stratify: bool = True,
        output_dir: str = "data/processed/"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Convenience method to split data and save in one step.
        
        Args:
            df: Input DataFrame containing features and target
            target_column: Name of the target variable column (default: 'is_canceled')
            stratify: Whether to stratify split on target variable (default: True)
            output_dir: Directory to save the datasets (default: 'data/processed/')
        
        Returns:
            Tuple containing:
                - X_train: Training features
                - X_test: Testing features
                - y_train: Training labels
                - y_test: Testing labels
        """
        logger.info("Starting split_and_save operation...")
        
        # Split the data
        X_train, X_test, y_train, y_test = self.split_data(
            df=df,
            target_column=target_column,
            stratify=stratify
        )
        
        # Save the splits
        self.save_splits(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            output_dir=output_dir
        )
        
        logger.info("split_and_save operation completed successfully")
        
        return X_train, X_test, y_train, y_test
