"""
Main Pipeline Execution Script

This script serves as the main entry point for the hotel cancellation prediction
training pipeline. It orchestrates data processing, model training, evaluation,
and hyperparameter optimization.

Usage:
    python run_pipeline.py                    # Run full pipeline
    python run_pipeline.py --stage data       # Run only data processing
    python run_pipeline.py --stage training   # Run only model training
    python run_pipeline.py --stage evaluation # Run only model evaluation
    python run_pipeline.py --stage optimization # Run only hyperparameter optimization
    python run_pipeline.py --config path/to/config.yaml  # Use custom config
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# Import data processing components
from src.data_processing.data_loader import DataLoader
from src.data_processing.data_cleaner import DataCleaner
from src.data_processing.feature_engineer import FeatureEngineer
from src.data_processing.data_splitter import DataSplitter

# Import modeling components
from src.modeling.model_trainer import ModelTrainer
from src.modeling.imbalance_handler import ImbalanceHandler
from src.modeling.hyperparameter_optimizer import HyperparameterOptimizer
from src.modeling.model_registry import ModelRegistry

# Import evaluation components
from src.evaluation.model_evaluator import ModelEvaluator
from src.evaluation.model_comparator import ModelComparator

# Import utilities
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.exceptions import DataLoadError, DataValidationError, ModelTrainingError

logger = get_logger(__name__)


def run_data_processing(config):
    """
    Execute data loading, cleaning, feature engineering, and splitting.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_engineer)
    
    Raises:
        DataLoadError: If data loading fails
        DataValidationError: If data validation fails
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: DATA PROCESSING")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # 1. Load data
        logger.info("Step 1.1: Loading raw data...")
        data_loader = DataLoader()
        raw_data_path = config['data']['raw_data_path']
        df = data_loader.load_csv(raw_data_path)
        
        # Validate schema
        data_loader.validate_schema(df, strict=False)
        
        # 2. Clean data
        logger.info("Step 1.2: Cleaning data...")
        data_cleaner = DataCleaner()
        df_clean = data_cleaner.clean_data(
            df,
            remove_duplicates=True,
            handle_missing=True,
            filter_invalid=True,
            missing_strategy=config['preprocessing'].get('numerical_imputation', 'auto'),
            missing_threshold=config['preprocessing'].get('missing_value_threshold', 0.3)
        )
        
        # Log cleaning stats
        cleaning_stats = data_cleaner.get_cleaning_stats()
        logger.info(f"Cleaning statistics: {cleaning_stats}")
        
        # 3. Feature engineering
        logger.info("Step 1.3: Engineering features...")
        feature_engineer = FeatureEngineer()
        
        # Get encoding configuration
        label_encode_cols = config['preprocessing']['categorical_encoding'].get('label_encode', [])
        onehot_encode_cols = config['preprocessing']['categorical_encoding'].get('onehot_encode', [])
        
        # Drop unnecessary columns
        features_to_drop = config['preprocessing'].get('features_to_drop', [])
        if features_to_drop:
            logger.info(f"Dropping columns: {features_to_drop}")
            df_clean = df_clean.drop(columns=[col for col in features_to_drop if col in df_clean.columns])
        
        # Apply feature engineering
        df_processed = feature_engineer.fit_transform(
            df_clean,
            label_encode_cols=label_encode_cols,
            onehot_encode_cols=onehot_encode_cols,
            scale_cols=None,  # Auto-detect numerical columns
            apply_log_transform=True,
            skewness_threshold=1.0
        )
        
        # Save fitted transformers
        logger.info("Saving fitted transformers...")
        feature_engineer.save_transformers(
            output_dir=config['data']['processed_data_path'],
            prefix="feature_engineer"
        )
        
        # 4. Split data
        logger.info("Step 1.4: Splitting data into train and test sets...")
        data_splitter = DataSplitter(
            test_size=config['data'].get('test_size', 0.2),
            random_state=config['data'].get('random_state', 42)
        )
        
        X_train, X_test, y_train, y_test = data_splitter.split_and_save(
            df=df_processed,
            target_column='is_canceled',
            stratify=True,
            output_dir=config['data']['processed_data_path']
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Data processing completed in {processing_time:.2f} seconds")
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, feature_engineer
    
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise


def run_training(config, X_train, y_train):
    """
    Train all models with cross-validation.
    
    Args:
        config: Configuration dictionary
        X_train: Training features
        y_train: Training labels
    
    Returns:
        dict: Dictionary of trained models and their CV scores
    
    Raises:
        ModelTrainingError: If model training fails
    """
    logger.info("=" * 80)
    logger.info("STAGE 2: MODEL TRAINING")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # 1. Check for class imbalance
        logger.info("Step 2.1: Checking for class imbalance...")
        imbalance_handler = ImbalanceHandler()
        imbalance_ratio = imbalance_handler.check_imbalance(y_train)
        
        imbalance_threshold = config['evaluation'].get('imbalance_threshold', 0.7)
        
        if imbalance_ratio > imbalance_threshold:
            logger.warning(f"Class imbalance detected: {imbalance_ratio:.2%}")
            logger.info("Applying SMOTE to balance classes...")
            X_train_balanced, y_train_balanced = imbalance_handler.apply_smote(X_train, y_train)
        else:
            logger.info(f"Class distribution is acceptable: {imbalance_ratio:.2%}")
            X_train_balanced, y_train_balanced = X_train, y_train
        
        # 2. Train all models
        logger.info("Step 2.2: Training all enabled models...")
        model_trainer = ModelTrainer(
            models_config=config['models'],
            cv_folds=config.get('hyperparameter_tuning', {}).get('cv_folds', 5)
        )
        
        training_results = model_trainer.train_all_models(X_train_balanced, y_train_balanced)
        
        # 3. Save trained models
        logger.info("Step 2.3: Saving trained models...")
        model_registry = ModelRegistry(models_dir=config.get('prediction', {}).get('model_path', 'models').replace('/best_model.pkl', ''))
        
        for model_name, result in training_results.items():
            if result['model'] is not None:
                model_registry.save_model(
                    model=result['model'],
                    model_name=model_name,
                    version="1.0.0",
                    metrics={
                        'cv_accuracy': result['cv_scores']['mean_accuracy'],
                        'cv_f1_score': result['cv_scores']['mean_f1'],
                        'cv_roc_auc': result['cv_scores']['mean_roc_auc']
                    },
                    hyperparameters=config['models'][model_name].get('params', {}),
                    additional_metadata={
                        'training_samples': len(X_train_balanced),
                        'num_features': X_train_balanced.shape[1]
                    }
                )
        
        training_time = time.time() - start_time
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        return training_results
    
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise


def run_evaluation(config, training_results, X_test, y_test):
    """
    Evaluate and compare all trained models.
    
    Args:
        config: Configuration dictionary
        training_results: Dictionary of trained models from run_training
        X_test: Test features
        y_test: Test labels
    
    Returns:
        tuple: (comparison_df, best_model_name, best_model)
    """
    logger.info("=" * 80)
    logger.info("STAGE 3: MODEL EVALUATION")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # 1. Extract trained models
        logger.info("Step 3.1: Extracting trained models...")
        models = {
            name: result['model']
            for name, result in training_results.items()
            if result['model'] is not None
        }
        
        if not models:
            logger.error("No trained models available for evaluation")
            raise ModelTrainingError("No trained models available")
        
        logger.info(f"Evaluating {len(models)} models: {list(models.keys())}")
        
        # 2. Compare models
        logger.info("Step 3.2: Comparing models on test set...")
        reports_dir = 'reports'
        if 'reports' in config and 'output_dir' in config['reports']:
            reports_dir = config['reports']['output_dir']
        model_comparator = ModelComparator(output_dir=reports_dir)
        
        comparison_df = model_comparator.compare_models(
            models=models,
            X_test=X_test,
            y_test=y_test
        )
        
        # 3. Rank models
        logger.info("Step 3.3: Ranking models by F1-score...")
        primary_metric = config['evaluation'].get('primary_metric', 'f1_score')
        ranked_df = model_comparator.rank_models(comparison_df, metric=primary_metric)
        
        # 4. Generate comparison report
        logger.info("Step 3.4: Generating comparison report...")
        report = model_comparator.generate_comparison_report(
            ranked_df,
            save_csv=True,
            csv_filename="model_comparison.csv"
        )
        
        # Print report to console
        print("\n" + report + "\n")
        
        # 5. Get best model
        logger.info("Step 3.5: Selecting best model...")
        best_model_name, best_model, best_metric_value = model_comparator.get_best_model(
            models=models,
            comparison_df=ranked_df,
            metric=primary_metric
        )
        
        logger.info(f"Best model: {best_model_name} with {primary_metric}={best_metric_value:.4f}")
        
        # 6. Save best model
        logger.info("Step 3.6: Saving best model...")
        model_registry = ModelRegistry(models_dir=config.get('prediction', {}).get('model_path', 'models').replace('/best_model.pkl', ''))
        
        # Get metrics for best model
        best_model_metrics = ranked_df[ranked_df['model_name'] == best_model_name].iloc[0].to_dict()
        best_model_metrics = {k: v for k, v in best_model_metrics.items() if k not in ['model_name', 'rank']}
        
        model_registry.save_model(
            model=best_model,
            model_name="best_model",
            version="1.0.0",
            metrics=best_model_metrics,
            hyperparameters=config['models'][best_model_name].get('params', {}),
            additional_metadata={
                'original_model_name': best_model_name,
                'test_samples': len(X_test),
                'num_features': X_test.shape[1]
            }
        )
        
        evaluation_time = time.time() - start_time
        logger.info(f"Model evaluation completed in {evaluation_time:.2f} seconds")
        
        return ranked_df, best_model_name, best_model
    
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise


def run_optimization(config, best_model_name, X_train, y_train, X_test, y_test):
    """
    Perform hyperparameter tuning on the best model.
    
    Args:
        config: Configuration dictionary
        best_model_name: Name of the best performing model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
    
    Returns:
        tuple: (optimized_model, best_params, best_score)
    """
    logger.info("=" * 80)
    logger.info("STAGE 4: HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # Check if hyperparameter tuning is enabled
        tuning_config = config.get('hyperparameter_tuning', {})
        if not tuning_config.get('enabled', False):
            logger.info("Hyperparameter tuning is disabled in configuration")
            return None, None, None
        
        # Check if parameter grid exists for the best model
        param_grids = tuning_config.get('param_grids', {})
        if best_model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {best_model_name}")
            return None, None, None
        
        logger.info(f"Step 4.1: Optimizing hyperparameters for {best_model_name}...")
        
        # Get base model configuration
        model_config = config['models'][best_model_name]
        base_params = model_config.get('params', {})
        
        # Create base model instance
        from src.modeling.model_trainer import ModelTrainer
        trainer = ModelTrainer(models_config=config['models'])
        base_model = trainer._create_model(best_model_name, base_params)
        
        # Initialize optimizer
        optimizer = HyperparameterOptimizer(
            param_grid=param_grids[best_model_name],
            search_method=tuning_config.get('method', 'randomized'),
            cv_folds=tuning_config.get('cv_folds', 5),
            n_iter=tuning_config.get('n_iter', 20),
            scoring=config['evaluation'].get('primary_metric', 'f1_weighted'),
            n_jobs=-1
        )
        
        # Perform optimization
        logger.info("Step 4.2: Running hyperparameter search...")
        search_object = optimizer.optimize(base_model, X_train, y_train, verbose=1)
        
        # Get best parameters and model
        best_params = optimizer.get_best_params()
        best_cv_score = optimizer.get_best_score()
        optimized_model = optimizer.get_best_estimator()
        
        # Evaluate optimized model on test set
        logger.info("Step 4.3: Evaluating optimized model on test set...")
        model_evaluator = ModelEvaluator()
        test_metrics = model_evaluator.evaluate_model(
            model=optimized_model,
            X_test=X_test,
            y_test=y_test,
            model_name=f"{best_model_name}_optimized"
        )
        
        logger.info(f"Optimized model test F1-score: {test_metrics['f1_score']:.4f}")
        
        # Check if optimization improved performance
        min_f1_threshold = tuning_config.get('min_f1_threshold', 0.80)
        if test_metrics['f1_score'] >= min_f1_threshold:
            logger.info(f"Optimized model meets F1-score threshold of {min_f1_threshold}")
        else:
            logger.warning(f"Optimized model F1-score ({test_metrics['f1_score']:.4f}) below threshold ({min_f1_threshold})")
        
        # Save optimized model
        logger.info("Step 4.4: Saving optimized model...")
        model_registry = ModelRegistry(models_dir=config.get('prediction', {}).get('model_path', 'models').replace('/best_model.pkl', ''))
        
        model_registry.save_model(
            model=optimized_model,
            model_name=f"{best_model_name}_optimized",
            version="2.0.0",
            metrics=test_metrics,
            hyperparameters=best_params,
            additional_metadata={
                'base_model': best_model_name,
                'cv_score': best_cv_score,
                'optimization_method': tuning_config.get('method', 'randomized'),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
        )
        
        # Save as best_model if it's better
        logger.info("Updating best_model with optimized version...")
        model_registry.save_model(
            model=optimized_model,
            model_name="best_model",
            version="2.0.0",
            metrics=test_metrics,
            hyperparameters=best_params,
            additional_metadata={
                'original_model_name': best_model_name,
                'optimized': True
            }
        )
        
        optimization_time = time.time() - start_time
        logger.info(f"Hyperparameter optimization completed in {optimization_time:.2f} seconds")
        
        return optimized_model, best_params, best_cv_score
    
    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}")
        raise


def save_final_report(config, pipeline_start_time, stages_completed):
    """
    Save a final pipeline execution report.
    
    Args:
        config: Configuration dictionary
        pipeline_start_time: Start time of the pipeline
        stages_completed: List of completed stages
    """
    logger.info("=" * 80)
    logger.info("GENERATING FINAL REPORT")
    logger.info("=" * 80)
    
    total_time = time.time() - pipeline_start_time
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("HOTEL CANCELLATION PREDICTION - PIPELINE EXECUTION REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Execution Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    report_lines.append("")
    report_lines.append("Stages Completed:")
    for i, stage in enumerate(stages_completed, 1):
        report_lines.append(f"  {i}. {stage}")
    report_lines.append("")
    report_lines.append("Configuration:")
    report_lines.append(f"  Data Path: {config['data']['raw_data_path']}")
    report_lines.append(f"  Test Size: {config['data'].get('test_size', 0.2)}")
    report_lines.append(f"  Random State: {config['data'].get('random_state', 42)}")
    report_lines.append(f"  Models Enabled: {[name for name, cfg in config['models'].items() if cfg.get('enabled', False)]}")
    report_lines.append("")
    report_lines.append("=" * 80)
    
    report = "\n".join(report_lines)
    
    # Print to console
    print("\n" + report + "\n")
    
    # Save to file
    reports_dir_path = 'reports'
    if 'reports' in config and 'output_dir' in config['reports']:
        reports_dir_path = config['reports']['output_dir']
    reports_dir = Path(reports_dir_path)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = reports_dir / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Final report saved to {report_path}")


def main():
    """
    Main entry point for the pipeline execution script.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Hotel Cancellation Prediction - Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                          # Run full pipeline
  python run_pipeline.py --stage data             # Run only data processing
  python run_pipeline.py --stage training         # Run only training
  python run_pipeline.py --stage evaluation       # Run only evaluation
  python run_pipeline.py --stage optimization     # Run only optimization
  python run_pipeline.py --config custom_config.yaml  # Use custom config
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--stage',
        type=str,
        choices=['data', 'training', 'evaluation', 'optimization', 'all'],
        default='all',
        help='Pipeline stage to run (default: all)'
    )
    
    args = parser.parse_args()
    
    # Start pipeline
    pipeline_start_time = time.time()
    stages_completed = []
    
    try:
        logger.info("=" * 80)
        logger.info("HOTEL CANCELLATION PREDICTION - TRAINING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Configuration File: {args.config}")
        logger.info(f"Stage: {args.stage}")
        logger.info("=" * 80)
        
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)
        
        # Initialize variables
        X_train, X_test, y_train, y_test = None, None, None, None
        feature_engineer = None
        training_results = None
        best_model_name = None
        best_model = None
        
        # Stage 1: Data Processing
        if args.stage in ['data', 'all']:
            X_train, X_test, y_train, y_test, feature_engineer = run_data_processing(config)
            stages_completed.append("Data Processing")
        
        # Load processed data if starting from training stage
        if args.stage in ['training', 'evaluation', 'optimization'] and X_train is None:
            logger.info("Loading processed data from disk...")
            data_splitter = DataSplitter()
            X_train, X_test, y_train, y_test = data_splitter.load_splits(
                input_dir=config['data']['processed_data_path']
            )
        
        # Stage 2: Model Training
        if args.stage in ['training', 'all']:
            training_results = run_training(config, X_train, y_train)
            stages_completed.append("Model Training")
        
        # Load trained models if starting from evaluation stage
        if args.stage in ['evaluation', 'optimization'] and training_results is None:
            logger.info("Loading trained models from registry...")
            model_registry = ModelRegistry(models_dir=config.get('prediction', {}).get('model_path', 'models').replace('/best_model.pkl', ''))
            models_info = model_registry.list_models()
            
            training_results = {}
            for model_info in models_info:
                if model_info['model_name'] != 'best_model' and not model_info['model_name'].endswith('_optimized'):
                    model, metadata = model_registry.load_model(model_info['file_path'])
                    training_results[model_info['model_name']] = {
                        'model': model,
                        'cv_scores': None
                    }
        
        # Stage 3: Model Evaluation
        if args.stage in ['evaluation', 'all']:
            ranked_df, best_model_name, best_model = run_evaluation(
                config, training_results, X_test, y_test
            )
            stages_completed.append("Model Evaluation")
        
        # Load best model if starting from optimization stage
        if args.stage == 'optimization' and best_model is None:
            logger.info("Loading best model from registry...")
            model_registry = ModelRegistry(models_dir=config.get('prediction', {}).get('model_path', 'models').replace('/best_model.pkl', ''))
            models_info = model_registry.list_models()
            
            # Find the best model (highest F1-score)
            best_model_info = max(
                [m for m in models_info if m['model_name'] != 'best_model' and not m['model_name'].endswith('_optimized')],
                key=lambda x: x.get('metrics', {}).get('cv_f1_score', 0)
            )
            best_model_name = best_model_info['model_name']
            best_model, _ = model_registry.load_model(best_model_info['file_path'])
        
        # Stage 4: Hyperparameter Optimization
        if args.stage in ['optimization', 'all']:
            if best_model_name:
                run_optimization(config, best_model_name, X_train, y_train, X_test, y_test)
                stages_completed.append("Hyperparameter Optimization")
            else:
                logger.warning("No best model available for optimization")
        
        # Generate final report
        save_final_report(config, pipeline_start_time, stages_completed)
        
        logger.info("=" * 80)
        logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        return 0
    
    except KeyboardInterrupt:
        logger.warning("Pipeline execution interrupted by user")
        return 1
    
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())