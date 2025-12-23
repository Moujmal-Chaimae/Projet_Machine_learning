"""
Unit tests for modeling components.

This module contains comprehensive tests for ModelTrainer, ImbalanceHandler,
HyperparameterOptimizer, and ModelRegistry classes.
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.modeling.model_trainer import ModelTrainer
from src.modeling.imbalance_handler import ImbalanceHandler
from src.modeling.hyperparameter_optimizer import HyperparameterOptimizer
from src.modeling.model_registry import ModelRegistry
from src.utils.exceptions import ModelTrainingError


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_training_data():
    """Create sample training data for testing."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    
    return X, y


@pytest.fixture
def imbalanced_data():
    """Create imbalanced dataset for testing."""
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    # Create imbalanced target (90% class 0, 10% class 1)
    y = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    
    return X, y


@pytest.fixture
def models_config():
    """Create sample model configuration."""
    return {
        'logistic_regression': {
            'enabled': True,
            'params': {
                'max_iter': 100,
                'random_state': 42
            }
        },
        'random_forest': {
            'enabled': True,
            'params': {
                'n_estimators': 10,
                'max_depth': 5,
                'random_state': 42
            }
        },
        'xgboost': {
            'enabled': True,
            'params': {
                'n_estimators': 10,
                'max_depth': 3,
                'random_state': 42
            }
        }
    }


@pytest.fixture
def trained_model(sample_training_data):
    """Create a trained model for testing."""
    X, y = sample_training_data
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model


# ============================================================================
# ModelTrainer Tests
# ============================================================================

class TestModelTrainer:
    """Tests for ModelTrainer class."""
    
    def test_init(self, models_config):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(models_config, cv_folds=5)
        
        assert trainer.models_config == models_config
        assert trainer.cv_folds == 5
        assert isinstance(trainer.trained_models, dict)
    
    def test_create_model_logistic_regression(self, models_config):
        """Test creating logistic regression model."""
        trainer = ModelTrainer(models_config)
        params = models_config['logistic_regression']['params']
        
        model = trainer._create_model('logistic_regression', params)
        
        assert isinstance(model, LogisticRegression)
        assert model.max_iter == 100
    
    def test_create_model_random_forest(self, models_config):
        """Test creating random forest model."""
        trainer = ModelTrainer(models_config)
        params = models_config['random_forest']['params']
        
        model = trainer._create_model('random_forest', params)
        
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 10
    
    def test_create_model_xgboost(self, models_config):
        """Test creating XGBoost model."""
        trainer = ModelTrainer(models_config)
        params = models_config['xgboost']['params']
        
        model = trainer._create_model('xgboost', params)
        
        assert isinstance(model, XGBClassifier)
        assert model.n_estimators == 10
    
    def test_create_model_unsupported(self, models_config):
        """Test creating unsupported model type."""
        trainer = ModelTrainer(models_config)
        
        with pytest.raises(ValueError, match="Unsupported model type"):
            trainer._create_model('unsupported_model', {})
    
    def test_train_model_success(self, models_config, sample_training_data):
        """Test successful model training."""
        X, y = sample_training_data
        trainer = ModelTrainer(models_config, cv_folds=3)
        
        model, cv_scores = trainer.train_model('logistic_regression', X, y)
        
        assert model is not None
        assert isinstance(cv_scores, dict)
        assert 'mean_accuracy' in cv_scores
        assert 'mean_f1' in cv_scores
        assert cv_scores['cv_folds'] == 3
    
    def test_train_model_not_enabled(self, models_config, sample_training_data):
        """Test training a disabled model."""
        X, y = sample_training_data
        models_config['logistic_regression']['enabled'] = False
        trainer = ModelTrainer(models_config)
        
        with pytest.raises(ValueError, match="not enabled"):
            trainer.train_model('logistic_regression', X, y)
    
    def test_train_model_not_in_config(self, models_config, sample_training_data):
        """Test training a model not in configuration."""
        X, y = sample_training_data
        trainer = ModelTrainer(models_config)
        
        with pytest.raises(ValueError, match="not found in configuration"):
            trainer.train_model('nonexistent_model', X, y)
    
    def test_apply_cross_validation(self, models_config, sample_training_data):
        """Test cross-validation application."""
        X, y = sample_training_data
        trainer = ModelTrainer(models_config, cv_folds=3)
        model = LogisticRegression(random_state=42)
        
        cv_scores = trainer.apply_cross_validation(model, X, y)
        
        assert isinstance(cv_scores, dict)
        assert 'mean_accuracy' in cv_scores
        assert 'std_accuracy' in cv_scores
        assert 'mean_precision' in cv_scores
        assert 'mean_recall' in cv_scores
        assert 'mean_f1' in cv_scores
        assert 'mean_roc_auc' in cv_scores
        assert 0 <= cv_scores['mean_accuracy'] <= 1
    
    def test_train_all_models(self, models_config, sample_training_data):
        """Test training all enabled models."""
        X, y = sample_training_data
        trainer = ModelTrainer(models_config, cv_folds=2)
        
        results = trainer.train_all_models(X, y)
        
        assert isinstance(results, dict)
        assert len(results) == 3  # All three models enabled
        
        for model_name, result in results.items():
            assert 'model' in result
            assert 'cv_scores' in result
            if result['model'] is not None:
                assert result['cv_scores'] is not None
    
    def test_get_trained_model(self, models_config, sample_training_data):
        """Test retrieving a trained model."""
        X, y = sample_training_data
        trainer = ModelTrainer(models_config, cv_folds=2)
        trainer.train_model('logistic_regression', X, y)
        
        model = trainer.get_trained_model('logistic_regression')
        
        assert model is not None
        assert isinstance(model, LogisticRegression)
    
    def test_get_trained_model_not_trained(self, models_config):
        """Test retrieving a model that hasn't been trained."""
        trainer = ModelTrainer(models_config)
        
        with pytest.raises(KeyError, match="has not been trained"):
            trainer.get_trained_model('logistic_regression')


# ============================================================================
# ImbalanceHandler Tests
# ============================================================================

class TestImbalanceHandler:
    """Tests for ImbalanceHandler class."""
    
    def test_init(self):
        """Test ImbalanceHandler initialization."""
        handler = ImbalanceHandler()
        assert handler is not None
    
    def test_check_imbalance_balanced(self, sample_training_data):
        """Test imbalance check with balanced data."""
        _, y = sample_training_data
        handler = ImbalanceHandler()
        
        ratio = handler.check_imbalance(y)
        
        assert isinstance(ratio, float)
        assert 0 <= ratio <= 1
        assert ratio < 0.7  # Should be relatively balanced
    
    def test_check_imbalance_imbalanced(self, imbalanced_data):
        """Test imbalance check with imbalanced data."""
        _, y = imbalanced_data
        handler = ImbalanceHandler()
        
        ratio = handler.check_imbalance(y)
        
        assert isinstance(ratio, float)
        assert ratio > 0.7  # Should detect imbalance
    
    def test_apply_smote_basic(self, imbalanced_data):
        """Test SMOTE application."""
        X, y = imbalanced_data
        handler = ImbalanceHandler()
        
        X_resampled, y_resampled = handler.apply_smote(X, y)
        
        # Check that minority class is oversampled
        assert len(X_resampled) >= len(X)
        assert len(y_resampled) >= len(y)
        
        # Check that classes are more balanced
        original_ratio = y.mean()
        resampled_ratio = y_resampled.mean()
        assert abs(resampled_ratio - 0.5) < abs(original_ratio - 0.5)
    
    def test_apply_smote_reproducibility(self, imbalanced_data):
        """Test SMOTE reproducibility with same random state."""
        X, y = imbalanced_data
        handler1 = ImbalanceHandler(random_state=42)
        handler2 = ImbalanceHandler(random_state=42)
        
        X_res1, y_res1 = handler1.apply_smote(X, y)
        X_res2, y_res2 = handler2.apply_smote(X, y)
        
        # Results should be identical with same random state
        assert np.array_equal(X_res1, X_res2)
        assert np.array_equal(y_res1, y_res2)
    
    def test_apply_smote_increases_minority_class(self, imbalanced_data):
        """Test that SMOTE increases minority class samples."""
        X, y = imbalanced_data
        handler = ImbalanceHandler()
        
        original_minority_count = np.sum(y == 1)
        X_resampled, y_resampled = handler.apply_smote(X, y)
        resampled_minority_count = np.sum(y_resampled == 1)
        
        assert resampled_minority_count > original_minority_count


# ============================================================================
# HyperparameterOptimizer Tests
# ============================================================================

class TestHyperparameterOptimizer:
    """Tests for HyperparameterOptimizer class."""
    
    def test_init_grid_search(self):
        """Test initialization with grid search."""
        param_grid = {'C': [0.1, 1.0], 'max_iter': [100, 200]}
        optimizer = HyperparameterOptimizer(
            param_grid=param_grid,
            search_method='grid',
            cv_folds=3
        )
        
        assert optimizer.param_grid == param_grid
        assert optimizer.search_method == 'grid'
        assert optimizer.cv_folds == 3
    
    def test_init_randomized_search(self):
        """Test initialization with randomized search."""
        param_grid = {'C': [0.1, 1.0, 10.0], 'max_iter': [100, 200, 300]}
        optimizer = HyperparameterOptimizer(
            param_grid=param_grid,
            search_method='randomized',
            n_iter=5
        )
        
        assert optimizer.search_method == 'randomized'
        assert optimizer.n_iter == 5
    
    def test_init_invalid_method(self):
        """Test initialization with invalid search method."""
        param_grid = {'C': [0.1, 1.0]}
        
        with pytest.raises(ValueError, match="search_method must be"):
            HyperparameterOptimizer(param_grid, search_method='invalid')
    
    def test_optimize_grid_search(self, sample_training_data):
        """Test optimization with grid search."""
        X, y = sample_training_data
        param_grid = {'C': [0.1, 1.0], 'max_iter': [50, 100]}
        
        optimizer = HyperparameterOptimizer(
            param_grid=param_grid,
            search_method='grid',
            cv_folds=2
        )
        
        model = LogisticRegression(random_state=42)
        search_object = optimizer.optimize(model, X, y, verbose=0)
        
        assert search_object is not None
        assert optimizer.best_params_ is not None
        assert optimizer.best_score_ is not None
        assert 'C' in optimizer.best_params_
        assert 'max_iter' in optimizer.best_params_
    
    def test_optimize_randomized_search(self, sample_training_data):
        """Test optimization with randomized search."""
        X, y = sample_training_data
        param_grid = {'C': [0.1, 1.0, 10.0], 'max_iter': [50, 100, 200]}
        
        optimizer = HyperparameterOptimizer(
            param_grid=param_grid,
            search_method='randomized',
            n_iter=3,
            cv_folds=2
        )
        
        model = LogisticRegression(random_state=42)
        search_object = optimizer.optimize(model, X, y, verbose=0)
        
        assert search_object is not None
        assert optimizer.best_params_ is not None
    
    def test_get_best_params(self, sample_training_data):
        """Test getting best parameters."""
        X, y = sample_training_data
        param_grid = {'C': [0.1, 1.0]}
        
        optimizer = HyperparameterOptimizer(param_grid, search_method='grid', cv_folds=2)
        model = LogisticRegression(random_state=42)
        optimizer.optimize(model, X, y, verbose=0)
        
        best_params = optimizer.get_best_params()
        
        assert isinstance(best_params, dict)
        assert 'C' in best_params
    
    def test_get_best_params_not_optimized(self):
        """Test getting best parameters before optimization."""
        param_grid = {'C': [0.1, 1.0]}
        optimizer = HyperparameterOptimizer(param_grid)
        
        with pytest.raises(RuntimeError, match="No optimization results"):
            optimizer.get_best_params()
    
    def test_get_best_score(self, sample_training_data):
        """Test getting best score."""
        X, y = sample_training_data
        param_grid = {'C': [0.1, 1.0]}
        
        optimizer = HyperparameterOptimizer(param_grid, search_method='grid', cv_folds=2)
        model = LogisticRegression(random_state=42)
        optimizer.optimize(model, X, y, verbose=0)
        
        best_score = optimizer.get_best_score()
        
        assert isinstance(best_score, float)
        assert 0 <= best_score <= 1
    
    def test_get_best_estimator(self, sample_training_data):
        """Test getting best estimator."""
        X, y = sample_training_data
        param_grid = {'C': [0.1, 1.0]}
        
        optimizer = HyperparameterOptimizer(param_grid, search_method='grid', cv_folds=2)
        model = LogisticRegression(random_state=42)
        optimizer.optimize(model, X, y, verbose=0)
        
        best_estimator = optimizer.get_best_estimator()
        
        assert best_estimator is not None
        assert isinstance(best_estimator, LogisticRegression)
    
    def test_get_cv_results(self, sample_training_data):
        """Test getting CV results."""
        X, y = sample_training_data
        param_grid = {'C': [0.1, 1.0]}
        
        optimizer = HyperparameterOptimizer(param_grid, search_method='grid', cv_folds=2)
        model = LogisticRegression(random_state=42)
        optimizer.optimize(model, X, y, verbose=0)
        
        cv_results = optimizer.get_cv_results()
        
        assert isinstance(cv_results, dict)
        assert 'params' in cv_results
        assert 'mean_test_score' in cv_results


# ============================================================================
# ModelRegistry Tests
# ============================================================================

class TestModelRegistry:
    """Tests for ModelRegistry class."""
    
    def test_init(self):
        """Test ModelRegistry initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(models_dir=temp_dir)
            
            assert registry.models_dir == Path(temp_dir)
            assert registry.models_dir.exists()
    
    def test_save_model_basic(self, trained_model):
        """Test basic model saving."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(models_dir=temp_dir)
            
            model_path = registry.save_model(
                model=trained_model,
                model_name='test_model',
                version='1.0.0'
            )
            
            assert os.path.exists(model_path)
            assert 'test_model_v1.0.0.pkl' in model_path
    
    def test_save_model_with_metadata(self, trained_model):
        """Test saving model with metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(models_dir=temp_dir)
            
            metrics = {'accuracy': 0.85, 'f1_score': 0.82}
            hyperparams = {'C': 1.0, 'max_iter': 100}
            
            model_path = registry.save_model(
                model=trained_model,
                model_name='test_model',
                version='1.0.0',
                metrics=metrics,
                hyperparameters=hyperparams
            )
            
            # Check that metadata file was created
            metadata_path = model_path.replace('.pkl', '_metadata.json')
            assert os.path.exists(metadata_path)
    
    def test_load_model_basic(self, trained_model):
        """Test basic model loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(models_dir=temp_dir)
            
            # Save model
            model_path = registry.save_model(
                model=trained_model,
                model_name='test_model',
                version='1.0.0'
            )
            
            # Load model
            loaded_model, metadata = registry.load_model(model_path)
            
            assert loaded_model is not None
            assert isinstance(loaded_model, LogisticRegression)
            assert isinstance(metadata, dict)
    
    def test_load_model_not_found(self):
        """Test loading non-existent model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(models_dir=temp_dir)
            
            with pytest.raises(ModelTrainingError, match="not found"):
                registry.load_model('nonexistent_model.pkl')
    
    def test_list_models_empty(self):
        """Test listing models when registry is empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(models_dir=temp_dir)
            
            models_info = registry.list_models()
            
            assert isinstance(models_info, list)
            assert len(models_info) == 0
    
    def test_list_models_with_models(self, trained_model):
        """Test listing models with saved models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(models_dir=temp_dir)
            
            # Save multiple models
            registry.save_model(trained_model, 'model1', '1.0.0')
            registry.save_model(trained_model, 'model2', '1.0.0')
            
            models_info = registry.list_models()
            
            assert len(models_info) == 2
            assert all('model_name' in info for info in models_info)
            assert all('version' in info for info in models_info)
    
    def test_get_best_model(self, trained_model):
        """Test getting best model by metric."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(models_dir=temp_dir)
            
            # Save models with different metrics
            registry.save_model(
                trained_model, 'model1', '1.0.0',
                metrics={'f1_score': 0.75}
            )
            registry.save_model(
                trained_model, 'model2', '1.0.0',
                metrics={'f1_score': 0.85}
            )
            
            best_model, metadata = registry.get_best_model(metric='f1_score')
            
            assert best_model is not None
            assert metadata['metrics']['f1_score'] == 0.85
    
    def test_get_best_model_no_models(self):
        """Test getting best model when no models exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(models_dir=temp_dir)
            
            result = registry.get_best_model()
            
            assert result is None
    
    def test_delete_model(self, trained_model):
        """Test deleting a model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(models_dir=temp_dir)
            
            # Save model
            model_path = registry.save_model(trained_model, 'test_model', '1.0.0')
            
            # Delete model
            success = registry.delete_model(model_path)
            
            assert success is True
            assert not os.path.exists(model_path)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=src.modeling', '--cov-report=term-missing'])
