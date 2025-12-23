"""
Unit tests for data processing components.

This module contains comprehensive tests for DataLoader, DataCleaner,
FeatureEngineer, and DataSplitter classes.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from pathlib import Path

from src.data_processing.data_loader import DataLoader
from src.data_processing.data_cleaner import DataCleaner
from src.data_processing.feature_engineer import FeatureEngineer
from src.data_processing.data_splitter import DataSplitter
from src.utils.exceptions import DataLoadError, DataValidationError


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_hotel_data():
    """Create a sample hotel booking DataFrame for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'hotel': np.random.choice(['Resort Hotel', 'City Hotel'], n_samples),
        'is_canceled': np.random.choice([0, 1], n_samples),
        'lead_time': np.random.randint(0, 365, n_samples),
        'arrival_date_year': np.random.choice([2015, 2016, 2017], n_samples),
        'arrival_date_month': np.random.choice(['January', 'February', 'March'], n_samples),
        'arrival_date_week_number': np.random.randint(1, 53, n_samples),
        'arrival_date_day_of_month': np.random.randint(1, 32, n_samples),
        'stays_in_weekend_nights': np.random.randint(0, 5, n_samples),
        'stays_in_week_nights': np.random.randint(0, 10, n_samples),
        'adults': np.random.randint(1, 4, n_samples),
        'children': np.random.choice([0.0, 1.0, 2.0], n_samples),
        'babies': np.random.randint(0, 2, n_samples),
        'meal': np.random.choice(['BB', 'HB', 'FB', 'SC'], n_samples),
        'country': np.random.choice(['PRT', 'GBR', 'USA', 'ESP'], n_samples),
        'market_segment': np.random.choice(['Direct', 'Corporate', 'Online TA'], n_samples),
        'distribution_channel': np.random.choice(['Direct', 'Corporate', 'TA/TO'], n_samples),
        'is_repeated_guest': np.random.choice([0, 1], n_samples),
        'previous_cancellations': np.random.randint(0, 3, n_samples),
        'previous_bookings_not_canceled': np.random.randint(0, 5, n_samples),
        'reserved_room_type': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'assigned_room_type': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'booking_changes': np.random.randint(0, 5, n_samples),
        'deposit_type': np.random.choice(['No Deposit', 'Refundable', 'Non Refund'], n_samples),
        'days_in_waiting_list': np.random.randint(0, 10, n_samples),
        'customer_type': np.random.choice(['Transient', 'Contract', 'Group'], n_samples),
        'adr': np.random.uniform(50, 300, n_samples),
        'required_car_parking_spaces': np.random.randint(0, 2, n_samples),
        'total_of_special_requests': np.random.randint(0, 5, n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_with_issues(sample_hotel_data):
    """Create sample data with duplicates, missing values, and invalid records."""
    df = sample_hotel_data.copy()
    
    # Add duplicates
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    
    # Add missing values
    df.loc[10:15, 'children'] = np.nan
    df.loc[20:25, 'country'] = np.nan
    
    # Add invalid records (zero guests)
    df.loc[30:32, 'adults'] = 0
    df.loc[30:32, 'children'] = 0
    df.loc[30:32, 'babies'] = 0
    
    return df


@pytest.fixture
def temp_csv_file(sample_hotel_data):
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_hotel_data.to_csv(f.name, index=False)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


# ============================================================================
# DataLoader Tests
# ============================================================================

class TestDataLoader:
    """Tests for DataLoader class."""
    
    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader is not None
        assert hasattr(loader, 'REQUIRED_COLUMNS')
    
    def test_load_csv_valid_file(self, temp_csv_file):
        """Test loading a valid CSV file."""
        loader = DataLoader()
        df = loader.load_csv(temp_csv_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'is_canceled' in df.columns
    
    def test_load_csv_file_not_found(self):
        """Test loading a non-existent file raises DataLoadError."""
        loader = DataLoader()
        
        with pytest.raises(DataLoadError, match="File not found"):
            loader.load_csv('nonexistent_file.csv')
    
    def test_load_csv_empty_file(self):
        """Test loading an empty CSV file raises DataLoadError."""
        loader = DataLoader()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('')
            temp_path = f.name
        
        try:
            with pytest.raises(DataLoadError):
                loader.load_csv(temp_path)
        finally:
            os.remove(temp_path)
    
    def test_validate_schema_valid_data(self, sample_hotel_data):
        """Test schema validation with valid data."""
        loader = DataLoader()
        result = loader.validate_schema(sample_hotel_data, strict=False)
        
        assert result is True
    
    def test_validate_schema_missing_columns(self, sample_hotel_data):
        """Test schema validation with missing required columns."""
        loader = DataLoader()
        df_incomplete = sample_hotel_data.drop(columns=['is_canceled'])
        
        with pytest.raises(DataValidationError, match="Missing required columns"):
            loader.validate_schema(df_incomplete)
    
    def test_validate_schema_empty_dataframe(self):
        """Test schema validation with empty DataFrame."""
        loader = DataLoader()
        df_empty = pd.DataFrame()
        
        with pytest.raises(DataValidationError, match="empty"):
            loader.validate_schema(df_empty)
    
    def test_get_data_summary(self, sample_hotel_data):
        """Test data summary generation."""
        loader = DataLoader()
        summary = loader.get_data_summary(sample_hotel_data)
        
        assert isinstance(summary, dict)
        assert 'shape' in summary
        assert 'num_rows' in summary
        assert 'cancellation_rate' in summary
        assert summary['num_rows'] == len(sample_hotel_data)


# ============================================================================
# DataCleaner Tests
# ============================================================================

class TestDataCleaner:
    """Tests for DataCleaner class."""
    
    def test_init(self):
        """Test DataCleaner initialization."""
        cleaner = DataCleaner()
        assert cleaner is not None
        assert hasattr(cleaner, 'cleaning_stats')
    
    def test_remove_duplicates(self, sample_data_with_issues):
        """Test duplicate removal."""
        cleaner = DataCleaner()
        initial_count = len(sample_data_with_issues)
        
        df_clean = cleaner.remove_duplicates(sample_data_with_issues)
        
        assert len(df_clean) < initial_count
        assert cleaner.cleaning_stats['duplicates_removed'] > 0
        assert len(df_clean) == len(df_clean.drop_duplicates())
    
    def test_remove_duplicates_no_duplicates(self, sample_hotel_data):
        """Test duplicate removal when no duplicates exist."""
        cleaner = DataCleaner()
        initial_count = len(sample_hotel_data)
        
        df_clean = cleaner.remove_duplicates(sample_hotel_data)
        
        assert len(df_clean) == initial_count
        assert cleaner.cleaning_stats['duplicates_removed'] == 0
    
    def test_remove_duplicates_invalid_input(self):
        """Test duplicate removal with invalid input."""
        cleaner = DataCleaner()
        
        with pytest.raises(DataValidationError):
            cleaner.remove_duplicates("not a dataframe")
    
    def test_handle_missing_values_auto_strategy(self, sample_data_with_issues):
        """Test missing value handling with auto strategy."""
        cleaner = DataCleaner()
        df_clean = cleaner.handle_missing_values(sample_data_with_issues, strategy='auto')
        
        # Check that missing values are reduced
        assert df_clean.isnull().sum().sum() <= sample_data_with_issues.isnull().sum().sum()
    
    def test_handle_missing_values_median_strategy(self, sample_data_with_issues):
        """Test missing value handling with median strategy."""
        cleaner = DataCleaner()
        df_clean = cleaner.handle_missing_values(sample_data_with_issues, strategy='median')
        
        # Numerical columns should have no missing values
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        assert df_clean[numerical_cols].isnull().sum().sum() == 0
    
    def test_handle_missing_values_drop_strategy(self, sample_data_with_issues):
        """Test missing value handling with drop strategy."""
        cleaner = DataCleaner()
        initial_count = len(sample_data_with_issues)
        
        df_clean = cleaner.handle_missing_values(sample_data_with_issues, strategy='drop')
        
        assert len(df_clean) < initial_count
        assert df_clean.isnull().sum().sum() == 0
    
    def test_handle_missing_values_invalid_strategy(self, sample_hotel_data):
        """Test missing value handling with invalid strategy."""
        cleaner = DataCleaner()
        
        with pytest.raises(DataValidationError, match="Invalid strategy"):
            cleaner.handle_missing_values(sample_hotel_data, strategy='invalid')
    
    def test_filter_invalid_records(self, sample_data_with_issues):
        """Test filtering of invalid records."""
        cleaner = DataCleaner()
        initial_count = len(sample_data_with_issues)
        
        df_clean = cleaner.filter_invalid_records(sample_data_with_issues)
        
        assert len(df_clean) < initial_count
        assert cleaner.cleaning_stats['invalid_records_removed'] > 0
        
        # Verify no records with zero guests remain
        total_guests = df_clean['adults'] + df_clean['children'].fillna(0) + df_clean['babies']
        assert (total_guests > 0).all()
    
    def test_filter_invalid_records_missing_columns(self, sample_hotel_data):
        """Test filtering with missing required columns."""
        cleaner = DataCleaner()
        df_incomplete = sample_hotel_data.drop(columns=['adults'])
        
        with pytest.raises(DataValidationError, match="Missing required columns"):
            cleaner.filter_invalid_records(df_incomplete)
    
    def test_clean_data_full_pipeline(self, sample_data_with_issues):
        """Test full cleaning pipeline."""
        cleaner = DataCleaner()
        initial_count = len(sample_data_with_issues)
        
        df_clean = cleaner.clean_data(
            sample_data_with_issues,
            remove_duplicates=True,
            handle_missing=True,
            filter_invalid=True
        )
        
        assert len(df_clean) < initial_count
        assert len(df_clean) > 0
        
        # Verify cleaning stats are populated
        stats = cleaner.get_cleaning_stats()
        assert 'duplicates_removed' in stats
        assert 'invalid_records_removed' in stats
    
    def test_get_cleaning_stats(self):
        """Test cleaning statistics retrieval."""
        cleaner = DataCleaner()
        stats = cleaner.get_cleaning_stats()
        
        assert isinstance(stats, dict)


# ============================================================================
# FeatureEngineer Tests
# ============================================================================

class TestFeatureEngineer:
    """Tests for FeatureEngineer class."""
    
    def test_init(self):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer()
        assert engineer is not None
        assert engineer.fitted is False
    
    def test_create_derived_features(self, sample_hotel_data):
        """Test creation of derived features."""
        engineer = FeatureEngineer()
        df_features = engineer.create_derived_features(sample_hotel_data)
        
        # Check that new features are created
        assert 'total_guests' in df_features.columns
        assert 'total_nights' in df_features.columns
        assert 'has_children' in df_features.columns
        assert 'is_long_stay' in df_features.columns
        assert 'price_per_night_per_guest' in df_features.columns
        
        # Verify calculations
        assert (df_features['total_guests'] >= 0).all()
        assert (df_features['total_nights'] >= 0).all()
        assert df_features['has_children'].isin([0, 1]).all()
    
    def test_create_derived_features_missing_columns(self):
        """Test derived feature creation with missing columns."""
        engineer = FeatureEngineer()
        df_incomplete = pd.DataFrame({'col1': [1, 2, 3]})
        
        with pytest.raises(DataValidationError, match="Missing required columns"):
            engineer.create_derived_features(df_incomplete)
    
    def test_encode_categorical_label_encoding(self, sample_hotel_data):
        """Test label encoding of categorical variables."""
        engineer = FeatureEngineer()
        label_cols = ['hotel', 'meal']
        
        df_encoded = engineer.encode_categorical(
            sample_hotel_data,
            label_encode_cols=label_cols,
            fit=True
        )
        
        # Check that columns are encoded
        for col in label_cols:
            assert pd.api.types.is_numeric_dtype(df_encoded[col])
        
        # Check that encoders are stored
        assert len(engineer.label_encoders) == len(label_cols)
    
    def test_encode_categorical_onehot_encoding(self, sample_hotel_data):
        """Test one-hot encoding of categorical variables."""
        engineer = FeatureEngineer()
        onehot_cols = ['market_segment']
        
        df_encoded = engineer.encode_categorical(
            sample_hotel_data,
            onehot_encode_cols=onehot_cols,
            fit=True
        )
        
        # Check that original column is removed
        assert 'market_segment' not in df_encoded.columns
        
        # Check that new columns are created
        assert len(engineer.onehot_feature_names) > 0
        assert any('market_segment_' in col for col in df_encoded.columns)
    
    def test_scale_numerical(self, sample_hotel_data):
        """Test numerical feature scaling."""
        engineer = FeatureEngineer()
        scale_cols = ['lead_time', 'adr']
        
        df_scaled = engineer.scale_numerical(
            sample_hotel_data,
            columns=scale_cols,
            fit=True
        )
        
        # Check that scaler is fitted
        assert engineer.scaler is not None
        
        # Check that scaled values have mean ~0 and std ~1
        for col in scale_cols:
            assert abs(df_scaled[col].mean()) < 0.1
            assert abs(df_scaled[col].std() - 1.0) < 0.1
    
    def test_scale_numerical_auto_detect(self, sample_hotel_data):
        """Test numerical scaling with auto-detection."""
        engineer = FeatureEngineer()
        
        df_scaled = engineer.scale_numerical(sample_hotel_data, columns=None, fit=True)
        
        assert engineer.scaler is not None
    
    def test_apply_transformations(self, sample_hotel_data):
        """Test log transformations for skewed features."""
        engineer = FeatureEngineer()
        
        df_transformed = engineer.apply_transformations(
            sample_hotel_data,
            skewness_threshold=1.0,
            fit=True
        )
        
        # Check that log-transformed columns are tracked
        assert isinstance(engineer.log_transformed_cols, list)
    
    def test_fit_transform(self, sample_hotel_data):
        """Test full fit_transform pipeline."""
        engineer = FeatureEngineer()
        
        df_transformed = engineer.fit_transform(
            sample_hotel_data,
            label_encode_cols=['hotel', 'meal'],
            onehot_encode_cols=['market_segment'],
            scale_cols=None,
            apply_log_transform=True
        )
        
        assert engineer.fitted is True
        assert len(engineer.feature_names) > 0
        assert df_transformed.shape[1] >= sample_hotel_data.shape[1]
    
    def test_transform_without_fit(self, sample_hotel_data):
        """Test transform without fitting first."""
        engineer = FeatureEngineer()
        
        with pytest.raises(DataValidationError, match="not fitted"):
            engineer.transform(sample_hotel_data)
    
    def test_save_and_load_transformers(self, sample_hotel_data):
        """Test saving and loading fitted transformers."""
        engineer = FeatureEngineer()
        
        # Fit transformers
        engineer.fit_transform(
            sample_hotel_data,
            label_encode_cols=['hotel'],
            scale_cols=['lead_time']
        )
        
        # Save transformers
        with tempfile.TemporaryDirectory() as temp_dir:
            engineer.save_transformers(temp_dir)
            
            # Create new engineer and load
            engineer2 = FeatureEngineer()
            engineer2.load_transformers(temp_dir)
            
            assert engineer2.fitted is True
            assert len(engineer2.label_encoders) == len(engineer.label_encoders)
    
    def test_get_feature_names(self, sample_hotel_data):
        """Test getting feature names after transformation."""
        engineer = FeatureEngineer()
        engineer.fit_transform(sample_hotel_data, label_encode_cols=['hotel'])
        
        feature_names = engineer.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0


# ============================================================================
# DataSplitter Tests
# ============================================================================

class TestDataSplitter:
    """Tests for DataSplitter class."""
    
    def test_init(self):
        """Test DataSplitter initialization."""
        splitter = DataSplitter(test_size=0.2, random_state=42)
        assert splitter.test_size == 0.2
        assert splitter.random_state == 42
    
    def test_split_data_basic(self, sample_hotel_data):
        """Test basic data splitting."""
        splitter = DataSplitter(test_size=0.2, random_state=42)
        
        X_train, X_test, y_train, y_test = splitter.split_data(sample_hotel_data)
        
        # Check shapes
        assert len(X_train) + len(X_test) == len(sample_hotel_data)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        
        # Check test size ratio
        test_ratio = len(X_test) / len(sample_hotel_data)
        assert abs(test_ratio - 0.2) < 0.05
    
    def test_split_data_with_stratification(self, sample_hotel_data):
        """Test data splitting with stratification."""
        splitter = DataSplitter(test_size=0.2, random_state=42)
        
        X_train, X_test, y_train, y_test = splitter.split_data(
            sample_hotel_data,
            stratify=True
        )
        
        # Check that class distributions are similar
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        
        assert abs(train_ratio - test_ratio) < 0.1
    
    def test_split_data_missing_target(self, sample_hotel_data):
        """Test splitting with missing target column."""
        splitter = DataSplitter()
        df_no_target = sample_hotel_data.drop(columns=['is_canceled'])
        
        with pytest.raises(DataValidationError, match="Target column"):
            splitter.split_data(df_no_target)
    
    def test_split_data_empty_dataframe(self):
        """Test splitting empty DataFrame."""
        splitter = DataSplitter()
        df_empty = pd.DataFrame()
        
        with pytest.raises(DataValidationError, match="empty"):
            splitter.split_data(df_empty)
    
    def test_save_and_load_splits(self, sample_hotel_data):
        """Test saving and loading data splits."""
        splitter = DataSplitter(test_size=0.2, random_state=42)
        
        X_train, X_test, y_train, y_test = splitter.split_data(sample_hotel_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save splits
            splitter.save_splits(X_train, X_test, y_train, y_test, output_dir=temp_dir)
            
            # Load splits
            X_train_loaded, X_test_loaded, y_train_loaded, y_test_loaded = splitter.load_splits(temp_dir)
            
            # Verify loaded data matches original
            assert X_train.shape == X_train_loaded.shape
            assert X_test.shape == X_test_loaded.shape
            assert len(y_train) == len(y_train_loaded)
            assert len(y_test) == len(y_test_loaded)
    
    def test_split_and_save(self, sample_hotel_data):
        """Test combined split and save operation."""
        splitter = DataSplitter(test_size=0.2, random_state=42)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            X_train, X_test, y_train, y_test = splitter.split_and_save(
                sample_hotel_data,
                output_dir=temp_dir
            )
            
            # Check that files were created
            assert os.path.exists(os.path.join(temp_dir, 'X_train.pkl'))
            assert os.path.exists(os.path.join(temp_dir, 'X_test.pkl'))
            assert os.path.exists(os.path.join(temp_dir, 'y_train.pkl'))
            assert os.path.exists(os.path.join(temp_dir, 'y_test.pkl'))


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=src.data_processing', '--cov-report=term-missing'])
