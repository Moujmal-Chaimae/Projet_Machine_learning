"""
Class imbalance handling for the hotel cancellation prediction system.
"""

import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ImbalanceHandler:
    """
    Handles class imbalance in training data using various techniques.
    
    This class provides methods to detect class imbalance and apply
    oversampling techniques like SMOTE to balance the dataset.
    """
    
    def __init__(self, imbalance_threshold: float = 0.7, random_state: int = 42):
        """
        Initialize the ImbalanceHandler.
        
        Args:
            imbalance_threshold: Ratio threshold to trigger resampling (default: 0.7)
                                If majority_class / total > threshold, apply SMOTE
            random_state: Random seed for reproducibility (default: 42)
        """
        self.imbalance_threshold = imbalance_threshold
        self.random_state = random_state
        logger.info(f"ImbalanceHandler initialized with threshold={imbalance_threshold}")
    
    def check_imbalance(self, y: np.ndarray) -> float:
        """
        Calculate the class distribution ratio.
        
        Args:
            y: Target variable array
        
        Returns:
            float: Ratio of majority class (value between 0.5 and 1.0)
        """
        if len(y) == 0:
            logger.warning("Empty target array provided to check_imbalance")
            return 0.0
        
        # Count class occurrences
        class_counts = Counter(y)
        total_samples = len(y)
        
        # Get majority class count
        majority_count = max(class_counts.values())
        majority_ratio = majority_count / total_samples
        
        logger.info(f"Class distribution: {dict(class_counts)}")
        logger.info(f"Majority class ratio: {majority_ratio:.3f}")
        
        return majority_ratio

    def apply_smote(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Apply SMOTE oversampling to balance the minority class.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target variable array (n_samples,)
        
        Returns:
            tuple: (X_resampled, y_resampled) - Balanced dataset
        """
        # Log original distribution
        original_counts = Counter(y)
        logger.info(f"Original class distribution: {dict(original_counts)}")
        logger.info(f"Original dataset shape: X={X.shape}, y={y.shape}")
        
        try:
            # Initialize SMOTE
            smote = SMOTE(random_state=self.random_state)
            
            # Apply SMOTE
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Log resampled distribution
            resampled_counts = Counter(y_resampled)
            logger.info(f"Resampled class distribution: {dict(resampled_counts)}")
            logger.info(f"Resampled dataset shape: X={X_resampled.shape}, y={y_resampled.shape}")
            
            # Calculate and log the change
            for class_label in original_counts.keys():
                original = original_counts[class_label]
                resampled = resampled_counts[class_label]
                change = resampled - original
                logger.info(f"Class {class_label}: {original} -> {resampled} (+{change} samples)")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"SMOTE resampling failed: {e}")
            logger.warning("Returning original data without resampling")
            return X, y
    
    def handle_imbalance(self, X: np.ndarray, y: np.ndarray, force: bool = False) -> tuple:
        """
        Check for imbalance and apply SMOTE if threshold is exceeded.
        
        This is the main method that combines imbalance detection and correction.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target variable array (n_samples,)
            force: If True, apply SMOTE regardless of threshold (default: False)
        
        Returns:
            tuple: (X_processed, y_processed) - Potentially resampled dataset
        """
        # Check imbalance ratio
        majority_ratio = self.check_imbalance(y)
        
        # Determine if resampling is needed
        needs_resampling = force or (majority_ratio > self.imbalance_threshold)
        
        if needs_resampling:
            logger.info(f"Class imbalance detected (ratio={majority_ratio:.3f} > threshold={self.imbalance_threshold})")
            logger.info("Applying SMOTE oversampling...")
            return self.apply_smote(X, y)
        else:
            logger.info(f"Class distribution is balanced (ratio={majority_ratio:.3f} <= threshold={self.imbalance_threshold})")
            logger.info("No resampling needed")
            return X, y
