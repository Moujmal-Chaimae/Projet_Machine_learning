"""
Hyperparameter Optimizer Module

This module provides the HyperparameterOptimizer class for performing hyperparameter
tuning using GridSearchCV or RandomizedSearchCV with cross-validation.
"""

import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HyperparameterOptimizer:
    """
    Performs hyperparameter optimization using grid search or randomized search.
    
    This class handles hyperparameter tuning for machine learning models using
    scikit-learn's GridSearchCV or RandomizedSearchCV with cross-validation.
    
    Attributes:
        param_grid (dict): Dictionary of hyperparameters to search
        search_method (str): Search method ('grid' or 'randomized')
        cv_folds (int): Number of folds for cross-validation
        n_iter (int): Number of iterations for randomized search
        scoring (str): Scoring metric for optimization
        n_jobs (int): Number of parallel jobs (-1 for all cores)
        best_params_ (dict): Best parameters found after optimization
        best_score_ (float): Best score achieved with optimal parameters
        search_results_ (dict): Complete search results
    """
    
    def __init__(self, param_grid, search_method='randomized', cv_folds=5, 
                 n_iter=20, scoring='f1_weighted', n_jobs=-1):
        """
        Initialize HyperparameterOptimizer with search configuration.
        
        Args:
            param_grid (dict): Dictionary of hyperparameters to search over.
                Example: {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30]}
            search_method (str): Search method to use ('grid' or 'randomized').
                Default: 'randomized'
            cv_folds (int): Number of folds for cross-validation. Default: 5
            n_iter (int): Number of parameter settings sampled for randomized search.
                Default: 20 (ignored for grid search)
            scoring (str): Scoring metric to optimize. Default: 'f1_weighted'
            n_jobs (int): Number of parallel jobs. Default: -1 (use all cores)
        
        Raises:
            ValueError: If search_method is not 'grid' or 'randomized'
        """
        if search_method not in ['grid', 'randomized']:
            raise ValueError(f"search_method must be 'grid' or 'randomized', got '{search_method}'")
        
        self.param_grid = param_grid
        self.search_method = search_method
        self.cv_folds = cv_folds
        self.n_iter = n_iter
        self.scoring = scoring
        self.n_jobs = n_jobs
        
        # Results attributes (populated after optimization)
        self.best_params_ = None
        self.best_score_ = None
        self.search_results_ = None
        self._search_object = None
        
        logger.info(f"HyperparameterOptimizer initialized with {search_method} search")
        logger.info(f"CV folds: {cv_folds}, Scoring: {scoring}")
        if search_method == 'randomized':
            logger.info(f"Randomized search iterations: {n_iter}")
        logger.debug(f"Parameter grid: {param_grid}")
    
    def optimize(self, model, X_train, y_train, verbose=1):
        """
        Perform hyperparameter search with cross-validation.
        
        This method executes either GridSearchCV or RandomizedSearchCV based on
        the configured search method. It logs progress and stores the best parameters.
        
        Args:
            model (object): Scikit-learn compatible model instance to optimize
            X_train (array-like): Training features
            y_train (array-like): Training labels
            verbose (int): Verbosity level for search (0=silent, 1=progress, 2=detailed).
                Default: 1
        
        Returns:
            object: Fitted search object (GridSearchCV or RandomizedSearchCV)
                with best estimator accessible via .best_estimator_
        
        Raises:
            Exception: If optimization fails
        """
        logger.info(f"Starting hyperparameter optimization using {self.search_method} search")
        logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        start_time = time.time()
        
        try:
            # Create search object based on method
            if self.search_method == 'grid':
                logger.info("Initializing GridSearchCV")
                self._search_object = GridSearchCV(
                    estimator=model,
                    param_grid=self.param_grid,
                    cv=self.cv_folds,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                    verbose=verbose,
                    return_train_score=True
                )
                total_combinations = self._calculate_grid_size()
                logger.info(f"Total parameter combinations to evaluate: {total_combinations}")
            else:  # randomized
                logger.info("Initializing RandomizedSearchCV")
                self._search_object = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=self.param_grid,
                    n_iter=self.n_iter,
                    cv=self.cv_folds,
                    scoring=self.scoring,
                    n_jobs=self.n_jobs,
                    verbose=verbose,
                    return_train_score=True,
                    random_state=42
                )
                logger.info(f"Sampling {self.n_iter} parameter combinations")
            
            # Perform search
            logger.info("Executing hyperparameter search (this may take a while)...")
            self._search_object.fit(X_train, y_train)
            
            # Store results
            self.best_params_ = self._search_object.best_params_
            self.best_score_ = self._search_object.best_score_
            self.search_results_ = self._search_object.cv_results_
            
            optimization_time = time.time() - start_time
            
            # Log results
            logger.info(f"Hyperparameter optimization completed in {optimization_time:.2f} seconds")
            logger.info(f"Best {self.scoring} score: {self.best_score_:.4f}")
            logger.info(f"Best parameters found:")
            for param, value in self.best_params_.items():
                logger.info(f"  {param}: {value}")
            
            # Log top 5 parameter combinations
            self._log_top_results(n=5)
            
            return self._search_object
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {str(e)}")
            raise
    
    def get_best_params(self):
        """
        Return optimal hyperparameters found during search.
        
        Returns:
            dict: Dictionary of best hyperparameters
        
        Raises:
            RuntimeError: If optimize() has not been called yet
        """
        if self.best_params_ is None:
            raise RuntimeError("No optimization results available. Call optimize() first.")
        
        logger.debug(f"Retrieving best parameters: {self.best_params_}")
        return self.best_params_
    
    def get_best_score(self):
        """
        Return the best score achieved during optimization.
        
        Returns:
            float: Best cross-validation score
        
        Raises:
            RuntimeError: If optimize() has not been called yet
        """
        if self.best_score_ is None:
            raise RuntimeError("No optimization results available. Call optimize() first.")
        
        return self.best_score_
    
    def get_best_estimator(self):
        """
        Return the best estimator (model trained with best parameters).
        
        Returns:
            object: Best estimator fitted on the entire training set
        
        Raises:
            RuntimeError: If optimize() has not been called yet
        """
        if self._search_object is None:
            raise RuntimeError("No optimization results available. Call optimize() first.")
        
        return self._search_object.best_estimator_
    
    def get_cv_results(self):
        """
        Return complete cross-validation results from the search.
        
        Returns:
            dict: Dictionary containing all CV results including:
                - params: list of parameter settings
                - mean_test_score: mean CV score for each parameter setting
                - std_test_score: standard deviation of CV scores
                - rank_test_score: rank of each parameter setting
        
        Raises:
            RuntimeError: If optimize() has not been called yet
        """
        if self.search_results_ is None:
            raise RuntimeError("No optimization results available. Call optimize() first.")
        
        return self.search_results_
    
    def _calculate_grid_size(self):
        """
        Calculate total number of parameter combinations in grid search.
        
        Returns:
            int: Total number of combinations
        """
        total = 1
        for param_values in self.param_grid.values():
            total *= len(param_values)
        return total
    
    def _log_top_results(self, n=5):
        """
        Log the top N parameter combinations and their scores.
        
        Args:
            n (int): Number of top results to log. Default: 5
        """
        if self.search_results_ is None:
            return
        
        # Get indices of top n results
        mean_scores = self.search_results_['mean_test_score']
        top_indices = mean_scores.argsort()[-n:][::-1]
        
        logger.info(f"Top {n} parameter combinations:")
        for i, idx in enumerate(top_indices, 1):
            score = mean_scores[idx]
            std = self.search_results_['std_test_score'][idx]
            params = self.search_results_['params'][idx]
            logger.info(f"  Rank {i}: Score={score:.4f} (+/- {std:.4f})")
            logger.debug(f"    Parameters: {params}")
