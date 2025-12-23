"""
Logging configuration for the hotel cancellation prediction system.
"""

import logging
import os
from pathlib import Path


def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    Configure and return a logger with file and console handlers.
    
    Args:
        name: Name of the logger (typically __name__ from calling module)
        log_file: Path to log file (optional, defaults to logs/hotel_cancellation.log)
        level: Logging level (default: INFO)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file is None:
        log_file = 'logs/hotel_cancellation.log'
    
    # Create logs directory if it doesn't exist
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str):
    """
    Get or create a logger with default configuration.
    
    Args:
        name: Name of the logger (typically __name__ from calling module)
    
    Returns:
        logging.Logger: Logger instance
    """
    return setup_logger(name)
