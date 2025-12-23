"""
Model comparison component for the hotel cancellation prediction system.

This module provides the ModelComparator class for comparing multiple trained models,
ranking them by performance metrics, and generating comparison reports.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from src.utils.logger import get_logger
from src.evaluation.model_evaluator import ModelEvaluator

logger = get_logger(__name__)


class ModelComparator:
    """
    Compares multiple machine learning models for hotel cancellation prediction.
    
    This class provides methods to evaluate multiple models on the same test set,
    rank them by performance metrics (primarily F1-score), and generate formatted
    comparison reports.
    """
    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize the ModelComparator.
        
        Args:
            output_dir: Directory to save comparison reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = ModelEvaluator()
        logger.info(f"ModelComparator initialized with output directory: {self.output_dir}")
    
    def compare_models(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """
        Evaluate all trained models on the test set and create comparison DataFrame.
        
        This method evaluates each model using the ModelEvaluator and compiles
        all metrics into a single DataFrame for easy comparison.
        
        Args:
            models: Dictionary mapping model names to trained model objects
            X_test: Test features
            y_test: Test labels
        
        Returns:
            DataFrame containing comparison metrics for all models
        """
        logger.info(f"Comparing {len(models)} models on test set")
        
        if not models:
            logger.warning("No models provided for comparison")
            return pd.DataFrame()
        
        comparison_results = []
        
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            
            try:
                # Evaluate the model
                metrics = self.evaluator.evaluate_model(
                    model=model,
                    X_test=X_test,
                    y_test=y_test,
                    model_name=model_name
                )
                
                # Remove confusion matrix from metrics for DataFrame
                metrics_for_df = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
                
                # Add model name to the results
                result = {'model_name': model_name}
                result.update(metrics_for_df)
                comparison_results.append(result)
                
                logger.info(f"Model {model_name} evaluated successfully")
                
            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {e}")
                # Add a row with NaN values for failed models
                comparison_results.append({
                    'model_name': model_name,
                    'accuracy': np.nan,
                    'precision': np.nan,
                    'recall': np.nan,
                    'f1_score': np.nan,
                    'roc_auc': np.nan
                })
        
        # Create DataFrame from results
        comparison_df = pd.DataFrame(comparison_results)
        
        # Reorder columns for better readability
        column_order = ['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        comparison_df = comparison_df[column_order]
        
        logger.info(f"Model comparison complete. {len(comparison_df)} models evaluated")
        
        return comparison_df
    
    def rank_models(
        self,
        comparison_df: pd.DataFrame,
        metric: str = 'f1_score',
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Sort models by the specified performance metric.
        
        By default, models are ranked by F1-score (primary metric) in descending order.
        
        Args:
            comparison_df: DataFrame containing model comparison results
            metric: Metric to use for ranking (default: 'f1_score')
            ascending: Sort order (default: False for descending)
        
        Returns:
            DataFrame sorted by the specified metric with rank column added
        """
        logger.info(f"Ranking models by {metric}")
        
        if comparison_df.empty:
            logger.warning("Empty comparison DataFrame provided for ranking")
            return comparison_df
        
        if metric not in comparison_df.columns:
            logger.error(f"Metric '{metric}' not found in comparison DataFrame")
            raise ValueError(f"Metric '{metric}' not found in comparison results")
        
        # Sort by the specified metric
        ranked_df = comparison_df.sort_values(by=metric, ascending=ascending).reset_index(drop=True)
        
        # Add rank column (1-indexed)
        ranked_df.insert(0, 'rank', range(1, len(ranked_df) + 1))
        
        logger.info(f"Models ranked by {metric}. Best model: {ranked_df.iloc[0]['model_name']}")
        
        return ranked_df
    
    def generate_comparison_report(
        self,
        comparison_df: pd.DataFrame,
        save_csv: bool = True,
        csv_filename: str = "model_comparison.csv"
    ) -> str:
        """
        Create a formatted comparison report as a string table.
        
        This method generates a human-readable text report showing all models
        and their performance metrics in a formatted table.
        
        Args:
            comparison_df: DataFrame containing model comparison results
            save_csv: Whether to save results to CSV file (default: True)
            csv_filename: Name of the CSV file to save (default: 'model_comparison.csv')
        
        Returns:
            Formatted string report
        """
        logger.info("Generating comparison report")
        
        if comparison_df.empty:
            logger.warning("Empty comparison DataFrame provided for report generation")
            return "No models to compare."
        
        # Save to CSV if requested
        if save_csv:
            csv_path = self.output_dir / csv_filename
            comparison_df.to_csv(csv_path, index=False, float_format='%.4f')
            logger.info(f"Comparison results saved to {csv_path}")
        
        # Generate formatted text report
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("MODEL COMPARISON REPORT")
        report_lines.append("=" * 100)
        report_lines.append("")
        
        # Add summary statistics
        report_lines.append("Summary:")
        report_lines.append(f"  Total models evaluated: {len(comparison_df)}")
        
        if 'rank' in comparison_df.columns:
            best_model = comparison_df.iloc[0]
            report_lines.append(f"  Best model: {best_model['model_name']}")
            report_lines.append(f"  Best F1-score: {best_model['f1_score']:.4f}")
        
        report_lines.append("")
        report_lines.append("-" * 100)
        report_lines.append("")
        
        # Create formatted table
        report_lines.append("Detailed Results:")
        report_lines.append("")
        
        # Format DataFrame as string with proper alignment
        df_string = comparison_df.to_string(index=False, float_format=lambda x: f'{x:.4f}')
        report_lines.append(df_string)
        
        report_lines.append("")
        report_lines.append("=" * 100)
        
        # Join all lines into a single report string
        report = "\n".join(report_lines)
        
        logger.info("Comparison report generated successfully")
        
        return report
    
    def get_best_model(
        self,
        models: Dict[str, Any],
        comparison_df: pd.DataFrame,
        metric: str = 'f1_score'
    ) -> tuple:
        """
        Get the best performing model based on the specified metric.
        
        Args:
            models: Dictionary mapping model names to trained model objects
            comparison_df: DataFrame containing model comparison results
            metric: Metric to use for selection (default: 'f1_score')
        
        Returns:
            Tuple of (best_model_name, best_model_object, best_metric_value)
        """
        logger.info(f"Selecting best model based on {metric}")
        
        if comparison_df.empty:
            logger.error("Cannot select best model from empty comparison DataFrame")
            raise ValueError("Comparison DataFrame is empty")
        
        if metric not in comparison_df.columns:
            logger.error(f"Metric '{metric}' not found in comparison DataFrame")
            raise ValueError(f"Metric '{metric}' not found in comparison results")
        
        # Find the row with the best metric value
        best_idx = comparison_df[metric].idxmax()
        best_row = comparison_df.loc[best_idx]
        
        best_model_name = best_row['model_name']
        best_metric_value = best_row[metric]
        best_model = models.get(best_model_name)
        
        if best_model is None:
            logger.error(f"Best model '{best_model_name}' not found in models dictionary")
            raise ValueError(f"Model '{best_model_name}' not found in provided models")
        
        logger.info(f"Best model selected: {best_model_name} with {metric}={best_metric_value:.4f}")
        
        return best_model_name, best_model, best_metric_value
