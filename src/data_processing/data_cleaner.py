"""
Data cleaning module for the hotel cancellation prediction system.
Handles duplicate removal, missing value imputation, and invalid record filtering.
"""

import pandas as pd
import numpy as np
from typing import Union, List
from src.utils.logger import get_logger
from src.utils.exceptions import DataValidationError

logger = get_logger(__name__)


class DataCleaner:
    """
    Cleans and validates hotel booking data.
    
    This class provides methods to:
    - Remove duplicate booking records
    - Handle missing values with various imputation strategies
    - Filter out invalid records (e.g., bookings with zero total guests)
    """
    
    def __init__(self):
        """Initialize the DataCleaner."""
        self.cleaning_stats = {}
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate booking records from the dataset.
        
        Args:
            df: Input DataFrame with potential duplicates
        
        Returns:
            pd.DataFrame: DataFrame with duplicates removed
        
        Raises:
            DataValidationError: If input is not a DataFrame or is empty
        """
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise DataValidationError("Input DataFrame is empty")
        
        initial_count = len(df)
        
        # Remove duplicates keeping the first occurrence
        df_clean = df.drop_duplicates(keep='first')
        
        duplicates_removed = initial_count - len(df_clean)
        
        self.cleaning_stats['duplicates_removed'] = duplicates_removed
        
        logger.info(f"Removed {duplicates_removed} duplicate records "
                   f"({duplicates_removed/initial_count*100:.2f}% of data)")
        logger.info(f"Remaining records: {len(df_clean)}")
        
        return df_clean
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        strategy: str = 'auto',
        threshold: float = 0.3
    ) -> pd.DataFrame:
        """
        Handle missing values using median/mode imputation strategies.
        
        Args:
            df: Input DataFrame with missing values
            strategy: Imputation strategy - 'auto', 'median', 'mode', or 'drop'
                     'auto' uses median for numerical and mode for categorical
            threshold: Maximum proportion of missing values allowed (0-1)
                      Columns exceeding this threshold are dropped
        
        Returns:
            pd.DataFrame: DataFrame with missing values handled
        
        Raises:
            DataValidationError: If input is invalid or strategy is unknown
        """
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise DataValidationError("Input DataFrame is empty")
        
        if not 0 <= threshold <= 1:
            raise DataValidationError("Threshold must be between 0 and 1")
        
        valid_strategies = ['auto', 'median', 'mode', 'drop']
        if strategy not in valid_strategies:
            raise DataValidationError(
                f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}"
            )
        
        df_clean = df.copy()
        initial_missing = df_clean.isnull().sum().sum()
        
        # Log initial missing value counts
        missing_by_column = df_clean.isnull().sum()
        columns_with_missing = missing_by_column[missing_by_column > 0]
        
        if len(columns_with_missing) > 0:
            logger.info(f"Found {initial_missing} missing values across "
                       f"{len(columns_with_missing)} columns")
            for col, count in columns_with_missing.items():
                pct = count / len(df_clean) * 100
                logger.debug(f"  {col}: {count} missing ({pct:.2f}%)")
        
        # Drop columns with too many missing values
        columns_to_drop = []
        for col in df_clean.columns:
            missing_pct = df_clean[col].isnull().sum() / len(df_clean)
            if missing_pct > threshold:
                columns_to_drop.append(col)
                logger.warning(
                    f"Dropping column '{col}' with {missing_pct*100:.2f}% missing values "
                    f"(exceeds threshold of {threshold*100:.2f}%)"
                )
        
        if columns_to_drop:
            df_clean = df_clean.drop(columns=columns_to_drop)
        
        # Handle remaining missing values based on strategy
        if strategy == 'drop':
            rows_before = len(df_clean)
            df_clean = df_clean.dropna()
            rows_dropped = rows_before - len(df_clean)
            logger.info(f"Dropped {rows_dropped} rows with missing values")
        
        else:  # auto, median, or mode
            for col in df_clean.columns:
                if df_clean[col].isnull().sum() > 0:
                    if strategy == 'auto':
                        # Use median for numerical, mode for categorical
                        if pd.api.types.is_numeric_dtype(df_clean[col]):
                            fill_value = df_clean[col].median()
                            method = 'median'
                        else:
                            fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else None
                            method = 'mode'
                    elif strategy == 'median':
                        if pd.api.types.is_numeric_dtype(df_clean[col]):
                            fill_value = df_clean[col].median()
                            method = 'median'
                        else:
                            # Fall back to mode for non-numeric columns
                            fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else None
                            method = 'mode (fallback)'
                    else:  # mode
                        fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else None
                        method = 'mode'
                    
                    if fill_value is not None:
                        missing_count = df_clean[col].isnull().sum()
                        df_clean[col] = df_clean[col].fillna(fill_value)
                        logger.debug(
                            f"Imputed {missing_count} missing values in '{col}' "
                            f"using {method} (value: {fill_value})"
                        )
        
        final_missing = df_clean.isnull().sum().sum()
        values_imputed = initial_missing - final_missing
        
        self.cleaning_stats['missing_values_handled'] = values_imputed
        self.cleaning_stats['columns_dropped'] = len(columns_to_drop)
        
        logger.info(f"Handled {values_imputed} missing values using '{strategy}' strategy")
        logger.info(f"Remaining missing values: {final_missing}")
        
        return df_clean
    
    def filter_invalid_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out invalid booking records.
        
        Removes records where:
        - Total guests (adults + children + babies) equals zero
        - Other business logic violations
        
        Args:
            df: Input DataFrame
        
        Returns:
            pd.DataFrame: DataFrame with invalid records removed
        
        Raises:
            DataValidationError: If required columns are missing
        """
        if not isinstance(df, pd.DataFrame):
            raise DataValidationError("Input must be a pandas DataFrame")
        
        if df.empty:
            raise DataValidationError("Input DataFrame is empty")
        
        # Check for required columns
        required_cols = ['adults', 'children', 'babies']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise DataValidationError(
                f"Missing required columns for validation: {missing_cols}"
            )
        
        df_clean = df.copy()
        initial_count = len(df_clean)
        
        # Calculate total guests
        df_clean['total_guests_temp'] = (
            df_clean['adults'].fillna(0) + 
            df_clean['children'].fillna(0) + 
            df_clean['babies'].fillna(0)
        )
        
        # Filter out records with zero total guests
        zero_guests_mask = df_clean['total_guests_temp'] == 0
        zero_guests_count = zero_guests_mask.sum()
        
        df_clean = df_clean[~zero_guests_mask]
        
        # Remove temporary column
        df_clean = df_clean.drop(columns=['total_guests_temp'])
        
        invalid_removed = initial_count - len(df_clean)
        
        self.cleaning_stats['invalid_records_removed'] = invalid_removed
        self.cleaning_stats['zero_guests_removed'] = zero_guests_count
        
        logger.info(f"Removed {invalid_removed} invalid records:")
        logger.info(f"  - {zero_guests_count} bookings with zero total guests")
        logger.info(f"Remaining records: {len(df_clean)}")
        
        return df_clean
    
    def clean_data(
        self, 
        df: pd.DataFrame,
        remove_duplicates: bool = True,
        handle_missing: bool = True,
        filter_invalid: bool = True,
        missing_strategy: str = 'auto',
        missing_threshold: float = 0.3
    ) -> pd.DataFrame:
        """
        Apply all cleaning operations in sequence.
        
        This is a convenience method that applies all cleaning steps:
        1. Remove duplicates
        2. Handle missing values
        3. Filter invalid records
        
        Args:
            df: Input DataFrame
            remove_duplicates: Whether to remove duplicate records
            handle_missing: Whether to handle missing values
            filter_invalid: Whether to filter invalid records
            missing_strategy: Strategy for handling missing values
            missing_threshold: Threshold for dropping columns with missing values
        
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        logger.info("Starting data cleaning pipeline...")
        logger.info(f"Initial dataset shape: {df.shape}")
        
        df_clean = df.copy()
        
        if remove_duplicates:
            df_clean = self.remove_duplicates(df_clean)
        
        if handle_missing:
            df_clean = self.handle_missing_values(
                df_clean, 
                strategy=missing_strategy,
                threshold=missing_threshold
            )
        
        if filter_invalid:
            df_clean = self.filter_invalid_records(df_clean)
        
        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        logger.info(f"Total records removed: {len(df) - len(df_clean)} "
                   f"({(len(df) - len(df_clean))/len(df)*100:.2f}%)")
        
        return df_clean
    
    def get_cleaning_stats(self) -> dict:
        """
        Get statistics about the cleaning operations performed.
        
        Returns:
            dict: Dictionary containing cleaning statistics
        """
        return self.cleaning_stats.copy()
