"""
Data loading and validation module for hotel booking data.
"""

import pandas as pd
import os
from typing import Optional
from src.utils.exceptions import DataLoadError, DataValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Handles loading and validation of hotel booking data from CSV files.
    """
    
    # Required columns for hotel booking dataset
    REQUIRED_COLUMNS = [
        'hotel', 'is_canceled', 'lead_time', 'arrival_date_year',
        'arrival_date_month', 'arrival_date_week_number', 'arrival_date_day_of_month',
        'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children',
        'babies', 'meal', 'country', 'market_segment', 'distribution_channel',
        'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
        'reserved_room_type', 'assigned_room_type', 'booking_changes',
        'deposit_type', 'days_in_waiting_list', 'customer_type', 'adr',
        'required_car_parking_spaces', 'total_of_special_requests'
    ]
    
    # Expected data types for validation
    EXPECTED_DTYPES = {
        'hotel': 'object',
        'is_canceled': 'int64',
        'lead_time': 'int64',
        'arrival_date_year': 'int64',
        'arrival_date_month': 'object',
        'arrival_date_week_number': 'int64',
        'arrival_date_day_of_month': 'int64',
        'stays_in_weekend_nights': 'int64',
        'stays_in_week_nights': 'int64',
        'adults': 'int64',
        'children': 'float64',  # Can have NaN
        'babies': 'int64',
        'meal': 'object',
        'country': 'object',
        'market_segment': 'object',
        'distribution_channel': 'object',
        'is_repeated_guest': 'int64',
        'previous_cancellations': 'int64',
        'previous_bookings_not_canceled': 'int64',
        'reserved_room_type': 'object',
        'assigned_room_type': 'object',
        'booking_changes': 'int64',
        'deposit_type': 'object',
        'days_in_waiting_list': 'int64',
        'customer_type': 'object',
        'adr': 'float64',
        'required_car_parking_spaces': 'int64',
        'total_of_special_requests': 'int64'
    }
    
    def __init__(self):
        """Initialize the DataLoader."""
        logger.info("DataLoader initialized")
    
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load hotel booking data from a CSV file with error handling.
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional arguments to pass to pd.read_csv()
        
        Returns:
            pd.DataFrame: Loaded dataset
        
        Raises:
            DataLoadError: If file cannot be loaded
        """
        logger.info(f"Attempting to load data from: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            raise DataLoadError(error_msg)
        
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 500:
            logger.warning(f"Large file detected ({file_size_mb:.2f} MB). Loading may take time.")
        
        try:
            # Load CSV file
            df = pd.read_csv(file_path, **kwargs)
            
            logger.info(f"Successfully loaded data with shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Check if dataframe is empty
            if df.empty:
                error_msg = "Dataset file is empty"
                logger.error(error_msg)
                raise DataLoadError(error_msg)
            
            return df
            
        except FileNotFoundError:
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            raise DataLoadError(error_msg)
        
        except pd.errors.EmptyDataError:
            error_msg = "Dataset file is empty"
            logger.error(error_msg)
            raise DataLoadError(error_msg)
        
        except pd.errors.ParserError as e:
            error_msg = f"Invalid CSV format: {str(e)}"
            logger.error(error_msg)
            raise DataLoadError(error_msg)
        
        except Exception as e:
            error_msg = f"Unexpected error loading data: {str(e)}"
            logger.error(error_msg)
            raise DataLoadError(error_msg)
    
    def validate_schema(self, df: pd.DataFrame, strict: bool = False) -> bool:
        """
        Validate that the dataframe has required columns and appropriate data types.
        
        Args:
            df: DataFrame to validate
            strict: If True, enforce exact data types. If False, only check column presence.
        
        Returns:
            bool: True if validation passes
        
        Raises:
            DataValidationError: If validation fails
        """
        logger.info("Starting schema validation")
        
        # Check if dataframe is empty
        if df.empty:
            error_msg = "Dataset is empty after loading"
            logger.error(error_msg)
            raise DataValidationError(error_msg)
        
        # Check for required columns
        missing_columns = set(self.REQUIRED_COLUMNS) - set(df.columns)
        
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            logger.error(error_msg)
            raise DataValidationError(error_msg)
        
        logger.info("All required columns are present")
        
        # Check target variable
        if 'is_canceled' not in df.columns:
            error_msg = "Target variable 'is_canceled' is missing"
            logger.error(error_msg)
            raise DataValidationError(error_msg)
        
        if df['is_canceled'].isnull().all():
            error_msg = "Target variable 'is_canceled' has all null values"
            logger.error(error_msg)
            raise DataValidationError(error_msg)
        
        # Check data types if strict mode
        if strict:
            logger.info("Performing strict data type validation")
            type_mismatches = []
            
            for col, expected_dtype in self.EXPECTED_DTYPES.items():
                if col in df.columns:
                    actual_dtype = str(df[col].dtype)
                    # Allow some flexibility (e.g., int64 vs int32, float64 vs float32)
                    if not self._is_compatible_dtype(actual_dtype, expected_dtype):
                        type_mismatches.append(
                            f"{col}: expected {expected_dtype}, got {actual_dtype}"
                        )
            
            if type_mismatches:
                error_msg = f"Data type mismatches: {', '.join(type_mismatches)}"
                logger.warning(error_msg)
                # Don't raise error, just warn in strict mode
        
        # Log summary statistics
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Number of rows: {len(df)}")
        logger.info(f"Number of columns: {len(df.columns)}")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check for target variable distribution
        if 'is_canceled' in df.columns:
            cancellation_rate = df['is_canceled'].mean()
            logger.info(f"Cancellation rate: {cancellation_rate:.2%}")
            
            if cancellation_rate == 0 or cancellation_rate == 1:
                logger.warning("Target variable has only one class. This may cause issues in training.")
        
        logger.info("Schema validation completed successfully")
        return True
    
    def _is_compatible_dtype(self, actual: str, expected: str) -> bool:
        """
        Check if actual dtype is compatible with expected dtype.
        
        Args:
            actual: Actual data type as string
            expected: Expected data type as string
        
        Returns:
            bool: True if compatible
        """
        # Exact match
        if actual == expected:
            return True
        
        # Integer types are compatible
        if 'int' in actual and 'int' in expected:
            return True
        
        # Float types are compatible
        if 'float' in actual and 'float' in expected:
            return True
        
        # Object types are compatible with string
        if actual == 'object' and expected == 'object':
            return True
        
        return False
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Generate a summary of the loaded dataset.
        
        Args:
            df: DataFrame to summarize
        
        Returns:
            dict: Summary statistics
        """
        summary = {
            'shape': df.shape,
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': list(df.columns),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        if 'is_canceled' in df.columns:
            summary['cancellation_rate'] = df['is_canceled'].mean()
            summary['class_distribution'] = df['is_canceled'].value_counts().to_dict()
        
        logger.info("Generated data summary")
        return summary
