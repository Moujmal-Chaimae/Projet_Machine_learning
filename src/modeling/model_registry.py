"""
Model Registry for saving, loading, and managing trained models.
"""

import os
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger
from src.utils.exceptions import ModelTrainingError

logger = get_logger(__name__)


class ModelRegistry:
    """
    Manages model persistence with versioning and metadata tracking.
    
    Handles saving trained models with metadata (version, metrics, hyperparameters,
    training date) and loading models for prediction or further analysis.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the ModelRegistry.
        
        Args:
            models_dir: Directory path where models will be saved
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ModelRegistry initialized with directory: {self.models_dir}")
    
    def save_model(
        self,
        model: Any,
        model_name: str,
        version: str = "1.0.0",
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a trained model with metadata to disk.
        
        Args:
            model: Trained model object (sklearn, xgboost, etc.)
            model_name: Name of the model (e.g., 'random_forest', 'xgboost')
            version: Version string (default: '1.0.0')
            metrics: Dictionary of evaluation metrics (accuracy, f1_score, etc.)
            hyperparameters: Dictionary of model hyperparameters
            additional_metadata: Any additional metadata to store
        
        Returns:
            str: Path to the saved model file
        
        Raises:
            ModelTrainingError: If model saving fails
        """
        try:
            # Create filename with version
            filename = f"{model_name}_v{version}.pkl"
            model_path = self.models_dir / filename
            
            # Prepare metadata
            metadata = {
                "model_name": model_name,
                "version": version,
                "training_date": datetime.now().isoformat(),
                "model_type": type(model).__name__,
                "metrics": metrics or {},
                "hyperparameters": hyperparameters or {},
            }
            
            # Add additional metadata if provided
            if additional_metadata:
                metadata.update(additional_metadata)
            
            # Create model package
            model_package = {
                "model": model,
                "metadata": metadata
            }
            
            # Save model package
            with open(model_path, 'wb') as f:
                pickle.dump(model_package, f)
            
            # Save metadata separately as JSON for easy inspection
            metadata_path = self.models_dir / f"{model_name}_v{version}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved successfully: {model_path}")
            logger.info(f"Metadata saved: {metadata_path}")
            
            return str(model_path)
        
        except Exception as e:
            error_msg = f"Failed to save model '{model_name}': {str(e)}"
            logger.error(error_msg)
            raise ModelTrainingError(error_msg)
    
    def load_model(self, model_path: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model file (can be relative or absolute)
        
        Returns:
            Tuple[Any, Dict]: Tuple of (model object, metadata dictionary)
        
        Raises:
            ModelTrainingError: If model loading fails
        """
        try:
            model_path = Path(model_path)
            
            # If path is not absolute and doesn't exist, try prepending models_dir
            if not model_path.is_absolute() and not model_path.exists():
                model_path = self.models_dir / model_path
            
            # Check if file exists
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model package
            with open(model_path, 'rb') as f:
                model_package = pickle.load(f)
            
            # Extract model and metadata
            if isinstance(model_package, dict) and "model" in model_package:
                model = model_package["model"]
                metadata = model_package.get("metadata", {})
            else:
                # Handle legacy models saved without metadata
                model = model_package
                metadata = {}
                logger.warning(f"Loaded model without metadata: {model_path}")
            
            logger.info(f"Model loaded successfully: {model_path}")
            
            return model, metadata
        
        except Exception as e:
            error_msg = f"Failed to load model from '{model_path}': {str(e)}"
            logger.error(error_msg)
            raise ModelTrainingError(error_msg)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available saved models with their metadata.
        
        Returns:
            List[Dict]: List of dictionaries containing model information
        """
        models_info = []
        
        try:
            # Find all .pkl files in models directory
            model_files = list(self.models_dir.glob("*.pkl"))
            
            if not model_files:
                logger.info("No models found in registry")
                return models_info
            
            for model_file in model_files:
                # Try to load metadata from JSON file first
                metadata_file = model_file.with_name(
                    model_file.stem + "_metadata.json"
                )
                
                if metadata_file.exists():
                    # Load metadata from JSON
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                else:
                    # Try to load metadata from model package
                    try:
                        with open(model_file, 'rb') as f:
                            model_package = pickle.load(f)
                        
                        if isinstance(model_package, dict) and "metadata" in model_package:
                            metadata = model_package["metadata"]
                        else:
                            metadata = {
                                "model_name": model_file.stem,
                                "version": "unknown",
                                "model_type": "unknown"
                            }
                    except Exception as e:
                        logger.warning(f"Could not load metadata for {model_file}: {e}")
                        metadata = {
                            "model_name": model_file.stem,
                            "version": "unknown",
                            "model_type": "unknown"
                        }
                
                # Add file path to metadata
                model_info = {
                    "file_path": str(model_file),
                    "file_name": model_file.name,
                    **metadata
                }
                
                models_info.append(model_info)
            
            # Sort by training date (most recent first)
            models_info.sort(
                key=lambda x: x.get("training_date", ""),
                reverse=True
            )
            
            logger.info(f"Found {len(models_info)} models in registry")
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
        
        return models_info
    
    def get_best_model(self, metric: str = "f1_score") -> Optional[Tuple[Any, Dict[str, Any]]]:
        """
        Get the best performing model based on a specific metric.
        
        Args:
            metric: Metric name to use for comparison (default: 'f1_score')
        
        Returns:
            Optional[Tuple]: Tuple of (model, metadata) or None if no models found
        """
        models_info = self.list_models()
        
        if not models_info:
            logger.warning("No models available in registry")
            return None
        
        # Filter models that have the specified metric
        models_with_metric = [
            m for m in models_info
            if metric in m.get("metrics", {})
        ]
        
        if not models_with_metric:
            logger.warning(f"No models found with metric '{metric}'")
            return None
        
        # Find model with best metric value
        best_model_info = max(
            models_with_metric,
            key=lambda x: x["metrics"][metric]
        )
        
        logger.info(
            f"Best model by {metric}: {best_model_info['model_name']} "
            f"(v{best_model_info['version']}) = {best_model_info['metrics'][metric]:.4f}"
        )
        
        # Load and return the best model
        return self.load_model(best_model_info["file_path"])
    
    def delete_model(self, model_path: str) -> bool:
        """
        Delete a model and its metadata from the registry.
        
        Args:
            model_path: Path to the model file to delete
        
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            # Handle relative paths
            if not os.path.isabs(model_path):
                model_path = self.models_dir / model_path
            else:
                model_path = Path(model_path)
            
            # Delete model file
            if model_path.exists():
                model_path.unlink()
                logger.info(f"Deleted model: {model_path}")
            
            # Delete metadata file if it exists
            metadata_path = model_path.with_name(
                model_path.stem + "_metadata.json"
            )
            if metadata_path.exists():
                metadata_path.unlink()
                logger.info(f"Deleted metadata: {metadata_path}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete model '{model_path}': {e}")
            return False
