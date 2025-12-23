"""
Model Trainer Module

This module provides the ModelTrainer class for training multiple classification models
with cross-validation support.
"""

import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """
    Trains multiple classification models with cross-validation.
    
    This class handles training of Logistic Regression, Random Forest, and XGBoost
    models based on configuration parameters. It supports cross-validation for
    model evaluation during training.
    
    Attributes:
        models_config (dict): Dictionary of model configurations from config file
        cv_folds (int): Number of folds for cross-validation (default: 5)
        trained_models (dict): Dictionary storing trained model objects
    """
    
    def __init__(self, models_config, cv_folds=5):
        """
        Initialize ModelTrainer with model configurations.
        
        Args:
            models_config (dict): Dictionary containing model configurations with structure:
                {
                    'logistic_regression': {'enabled': bool, 'params': dict},
                    'random_forest': {'enabled': bool, 'params': dict},
                    'xgboost': {'enabled': bool, 'params': dict}
                }
            cv_folds (int): Number of folds for cross-validation (default: 5)
        """
        self.models_config = models_config
        self.cv_folds = cv_folds
        self.trained_models = {}
        
        logger.info(f"ModelTrainer initialized with {cv_folds}-fold cross-validation")
        logger.debug(f"Model configurations: {models_config}")
    
    def _create_model(self, model_name, params):
        """
        Create a model instance based on model name and parameters.
        
        Args:
            model_name (str): Name of the model ('logistic_regression', 'random_forest', 'xgboost')
            params (dict): Model hyperparameters
            
        Returns:
            object: Instantiated model object
            
        Raises:
            ValueError: If model_name is not supported
        """
        if model_name == 'logistic_regression':
            return LogisticRegression(**params)
        elif model_name == 'random_forest':
            return RandomForestClassifier(**params)
        elif model_name == 'xgboost':
            return XGBClassifier(**params, eval_metric='logloss')
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
    
    def train_model(self, model_name, X_train, y_train):
        """
        Train a single model with cross-validation.
        
        This method trains a specified model on the training data and performs
        cross-validation to evaluate its performance during training.
        
        Args:
            model_name (str): Name of the model to train
            X_train (array-like): Training features
            y_train (array-like): Training labels
            
        Returns:
            tuple: (trained_model, cv_scores_dict)
                - trained_model: Fitted model object
                - cv_scores_dict: Dictionary containing cross-validation scores
                
        Raises:
            ValueError: If model is not enabled in configuration
        """
        if model_name not in self.models_config:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        
        model_config = self.models_config[model_name]
        
        if not model_config.get('enabled', False):
            raise ValueError(f"Model '{model_name}' is not enabled in configuration")
        
        logger.info(f"Starting training for {model_name}")
        start_time = time.time()
        
        # Create model instance
        params = model_config.get('params', {})
        model = self._create_model(model_name, params)
        
        # Perform cross-validation before final training
        logger.info(f"Performing {self.cv_folds}-fold cross-validation for {model_name}")
        cv_scores = self.apply_cross_validation(model, X_train, y_train)
        
        # Train final model on full training set
        logger.info(f"Training final {model_name} model on full training set")
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        logger.info(f"{model_name} training completed in {training_time:.2f} seconds")
        logger.info(f"{model_name} CV scores - Accuracy: {cv_scores['mean_accuracy']:.4f} (+/- {cv_scores['std_accuracy']:.4f})")
        
        # Store trained model
        self.trained_models[model_name] = model
        
        return model, cv_scores
    
    def apply_cross_validation(self, model, X, y):
        """
        Perform k-fold cross-validation and return mean scores.
        
        This method evaluates the model using cross-validation with multiple metrics
        including accuracy, precision, recall, F1-score, and ROC-AUC.
        
        Args:
            model (object): Model instance to evaluate
            X (array-like): Features
            y (array-like): Labels
            
        Returns:
            dict: Dictionary containing mean and std of cross-validation scores:
                {
                    'mean_accuracy': float,
                    'std_accuracy': float,
                    'mean_precision': float,
                    'std_precision': float,
                    'mean_recall': float,
                    'std_recall': float,
                    'mean_f1': float,
                    'std_f1': float,
                    'mean_roc_auc': float,
                    'std_roc_auc': float,
                    'cv_folds': int
                }
        """
        logger.debug(f"Running {self.cv_folds}-fold cross-validation")
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'f1': 'f1_weighted',
            'roc_auc': 'roc_auc'
        }
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, 
            X, 
            y, 
            cv=self.cv_folds,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1  # Use all available cores
        )
        
        # Calculate mean and std for each metric
        cv_scores = {
            'mean_accuracy': np.mean(cv_results['test_accuracy']),
            'std_accuracy': np.std(cv_results['test_accuracy']),
            'mean_precision': np.mean(cv_results['test_precision']),
            'std_precision': np.std(cv_results['test_precision']),
            'mean_recall': np.mean(cv_results['test_recall']),
            'std_recall': np.std(cv_results['test_recall']),
            'mean_f1': np.mean(cv_results['test_f1']),
            'std_f1': np.std(cv_results['test_f1']),
            'mean_roc_auc': np.mean(cv_results['test_roc_auc']),
            'std_roc_auc': np.std(cv_results['test_roc_auc']),
            'cv_folds': self.cv_folds
        }
        
        logger.debug(f"Cross-validation completed: {cv_scores}")
        
        return cv_scores
    
    def train_all_models(self, X_train, y_train):
        """
        Train all enabled models (Logistic Regression, Random Forest, XGBoost).
        
        This method iterates through all enabled models in the configuration
        and trains each one, storing the results.
        
        Args:
            X_train (array-like): Training features
            y_train (array-like): Training labels
            
        Returns:
            dict: Dictionary of trained models and their CV scores:
                {
                    'model_name': {
                        'model': trained_model_object,
                        'cv_scores': cv_scores_dict
                    }
                }
        """
        logger.info("Starting training for all enabled models")
        logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        results = {}
        total_start_time = time.time()
        
        for model_name, model_config in self.models_config.items():
            if model_config.get('enabled', False):
                try:
                    model, cv_scores = self.train_model(model_name, X_train, y_train)
                    results[model_name] = {
                        'model': model,
                        'cv_scores': cv_scores
                    }
                    logger.info(f"[SUCCESS] {model_name} training successful")
                except Exception as e:
                    logger.error(f"[FAILED] {model_name} training failed: {str(e)}")
                    results[model_name] = {
                        'model': None,
                        'cv_scores': None,
                        'error': str(e)
                    }
            else:
                logger.info(f"Skipping {model_name} (disabled in configuration)")
        
        total_time = time.time() - total_start_time
        logger.info(f"All models training completed in {total_time:.2f} seconds")
        logger.info(f"Successfully trained {len([r for r in results.values() if r['model'] is not None])} models")
        
        return results
    
    def get_trained_model(self, model_name):
        """
        Retrieve a trained model by name.
        
        Args:
            model_name (str): Name of the model to retrieve
            
        Returns:
            object: Trained model object
            
        Raises:
            KeyError: If model has not been trained yet
        """
        if model_name not in self.trained_models:
            raise KeyError(f"Model '{model_name}' has not been trained yet")
        
        return self.trained_models[model_name]
