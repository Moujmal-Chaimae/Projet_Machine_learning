"""
Model evaluation component for the hotel cancellation prediction system.

This module provides the ModelEvaluator class for calculating performance metrics,
generating confusion matrices, and creating visualization plots for model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)
from pathlib import Path
from typing import Dict, Tuple, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluates machine learning models for hotel cancellation prediction.
    
    This class provides methods to calculate various performance metrics,
    generate confusion matrices, and create visualization plots including
    ROC curves and precision-recall curves.
    """
    
    def __init__(self, output_dir: str = "reports/figures"):
        """
        Initialize the ModelEvaluator.
        
        Args:
            output_dir: Directory to save visualization plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ModelEvaluator initialized with output directory: {self.output_dir}")
    
    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Evaluate a trained model on test data and calculate all metrics.
        
        This method generates predictions, calculates performance metrics,
        creates confusion matrix, and generates ROC and precision-recall curves.
        
        Args:
            model: Trained model with predict() and predict_proba() methods
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model for logging and file naming
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        logger.info(f"Evaluating model: {model_name}")
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Generate probability predictions
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            logger.warning(f"Model {model_name} does not have predict_proba method. ROC-AUC will not be calculated.")
            y_proba = None
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_proba)
        
        # Generate confusion matrix
        cm = self.generate_confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Plot ROC curve if probabilities are available
        if y_proba is not None:
            self.plot_roc_curve(y_test, y_proba, model_name)
            self.plot_precision_recall_curve(y_test, y_proba, model_name)
        
        logger.info(f"Model {model_name} evaluation complete. Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics from predictions and probabilities.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities for positive class (optional)
        
        Returns:
            Dictionary containing accuracy, precision, recall, F1-score, and ROC-AUC
        """
        logger.debug("Calculating performance metrics")
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }
        
        # Calculate ROC-AUC if probabilities are provided
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except ValueError as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
                metrics['roc_auc'] = None
        else:
            metrics['roc_auc'] = None
        
        logger.debug(f"Metrics calculated: {metrics}")
        
        return metrics
    
    def generate_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Generate confusion matrix from true and predicted labels.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Confusion matrix as numpy array
        """
        logger.debug("Generating confusion matrix")
        
        cm = confusion_matrix(y_true, y_pred)
        
        logger.debug(f"Confusion matrix:\n{cm}")
        
        return cm
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "model"
    ) -> None:
        """
        Plot and save ROC curve visualization.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class
            model_name: Name of the model for file naming
        """
        logger.info(f"Plotting ROC curve for {model_name}")
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_path = self.output_dir / f"roc_curve_{model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to {output_path}")
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str = "model"
    ) -> None:
        """
        Plot and save precision-recall curve visualization.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities for positive class
            model_name: Name of the model for file naming
        """
        logger.info(f"Plotting precision-recall curve for {model_name}")
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        # Calculate baseline (proportion of positive class)
        baseline = np.sum(y_true) / len(y_true)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.axhline(y=baseline, color='navy', lw=2, linestyle='--',
                   label=f'Baseline (Positive Rate = {baseline:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # Save plot
        output_path = self.output_dir / f"precision_recall_curve_{model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Precision-recall curve saved to {output_path}")
    
    def plot_confusion_matrix_heatmap(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model",
        labels: list = None
    ) -> None:
        """
        Plot and save confusion matrix as a heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model for file naming
            labels: Class labels for display (default: ['Not Cancelled', 'Cancelled'])
        """
        logger.info(f"Plotting confusion matrix heatmap for {model_name}")
        
        # Generate confusion matrix
        cm = self.generate_confusion_matrix(y_true, y_pred)
        
        # Set default labels if not provided
        if labels is None:
            labels = ['Not Cancelled', 'Cancelled']
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        
        # Save plot
        output_path = self.output_dir / f"confusion_matrix_{model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix heatmap saved to {output_path}")
