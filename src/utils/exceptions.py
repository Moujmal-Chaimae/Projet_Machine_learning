"""
Custom exception classes for the hotel cancellation prediction system.
"""


class DataLoadError(Exception):
    """Raised when data cannot be loaded from a file."""
    pass


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


class ModelTrainingError(Exception):
    """Raised when model training fails."""
    pass


class PredictionError(Exception):
    """Raised when prediction fails."""
    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass
