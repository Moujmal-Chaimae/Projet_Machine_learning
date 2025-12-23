"""
Configuration loader utility for the hotel cancellation prediction system.

This module provides functions to load and validate YAML configuration files
with support for environment variable substitution.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from src.utils.exceptions import ConfigurationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file with error handling.
    
    Args:
        config_path: Path to the YAML configuration file
    
    Returns:
        Dictionary containing configuration settings
    
    Raises:
        ConfigurationError: If configuration file is invalid or missing
    """
    logger.info(f"Loading configuration from: {config_path}")
    
    config_path = Path(config_path)
    
    # Check if file exists
    if not config_path.exists():
        error_msg = f"Config file not found: {config_path}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg)
    
    try:
        # Load YAML file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            error_msg = f"Config file is empty: {config_path}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
        
        logger.info("Configuration loaded successfully")
        
        # Validate configuration
        validate_config(config)
        
        # Apply environment variable substitution
        config = substitute_env_vars(config)
        
        logger.debug(f"Configuration: {config}")
        
        return config
    
    except yaml.YAMLError as e:
        error_msg = f"Invalid YAML format in config file: {e}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg)
    
    except Exception as e:
        error_msg = f"Error loading configuration: {e}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg)


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate that configuration has all required sections.
    
    Args:
        config: Configuration dictionary to validate
    
    Returns:
        True if validation passes
    
    Raises:
        ConfigurationError: If required sections are missing
    """
    logger.debug("Validating configuration structure")
    
    # Required top-level sections
    required_sections = ["data", "models", "evaluation"]
    
    missing_sections = [
        section for section in required_sections
        if section not in config
    ]
    
    if missing_sections:
        error_msg = f"Missing required config sections: {missing_sections}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg)
    
    # Validate data section
    if "raw_data_path" not in config["data"]:
        error_msg = "Missing 'raw_data_path' in data configuration"
        logger.error(error_msg)
        raise ConfigurationError(error_msg)
    
    # Validate models section
    if not config["models"]:
        error_msg = "No models configured in 'models' section"
        logger.error(error_msg)
        raise ConfigurationError(error_msg)
    
    # Check if at least one model is enabled
    enabled_models = [
        name for name, cfg in config["models"].items()
        if cfg.get("enabled", False)
    ]
    
    if not enabled_models:
        logger.warning("No models are enabled in configuration")
    
    logger.info("Configuration validation passed")
    return True


def substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Substitute environment variables in configuration values.
    
    Supports ${VAR_NAME} syntax for environment variable substitution.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Configuration dictionary with environment variables substituted
    """
    logger.debug("Substituting environment variables in configuration")
    
    def _substitute_value(value):
        """Recursively substitute environment variables in values."""
        if isinstance(value, str):
            # Check for ${VAR_NAME} pattern
            if value.startswith("${") and value.endswith("}"):
                var_name = value[2:-1]
                env_value = os.environ.get(var_name)
                if env_value is not None:
                    logger.debug(f"Substituted ${{{var_name}}} with environment value")
                    return env_value
                else:
                    logger.warning(f"Environment variable ${{{var_name}}} not found, keeping original value")
                    return value
            return value
        elif isinstance(value, dict):
            return {k: _substitute_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_substitute_value(item) for item in value]
        else:
            return value
    
    return _substitute_value(config)


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation path.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the value (e.g., 'data.raw_data_path')
        default: Default value if key is not found
    
    Returns:
        Configuration value or default
    
    Example:
        >>> config = {'data': {'raw_data_path': 'data/raw/file.csv'}}
        >>> get_config_value(config, 'data.raw_data_path')
        'data/raw/file.csv'
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        logger.debug(f"Config key '{key_path}' not found, using default: {default}")
        return default
