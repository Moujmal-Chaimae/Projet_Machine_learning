"""
Input Validator for hotel booking predictions.

This module provides validation for booking data before making predictions,
ensuring all required fields are present, data types are correct, and values
are within acceptable ranges.
"""

from typing import Dict, Any, Tuple, List, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class InputValidator:
    """
    Validates booking data for prediction requests.
    
    This class checks that booking data contains all required fields,
    has correct data types, and values are within acceptable ranges.
    """
    
    # Define required fields for prediction (excluding target variable and dropped features)
    REQUIRED_FIELDS = [
        "hotel",
        "lead_time",
        "arrival_date_year",
        "arrival_date_month",
        "arrival_date_week_number",
        "arrival_date_day_of_month",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adults",
        "children",
        "babies",
        "meal",
        "country",
        "market_segment",
        "distribution_channel",
        "is_repeated_guest",
        "previous_cancellations",
        "previous_bookings_not_canceled",
        "reserved_room_type",
        "assigned_room_type",
        "booking_changes",
        "deposit_type",
        "days_in_waiting_list",
        "customer_type",
        "adr",
        "required_car_parking_spaces",
        "total_of_special_requests"
    ]
    
    # Optional fields that can be None/null
    OPTIONAL_FIELDS = [
        "agent",
        "company"
    ]
    
    # Define expected data types for each field
    FIELD_TYPES = {
        # String fields
        "hotel": str,
        "arrival_date_month": str,
        "meal": str,
        "country": str,
        "market_segment": str,
        "distribution_channel": str,
        "reserved_room_type": str,
        "assigned_room_type": str,
        "deposit_type": str,
        "customer_type": str,
        
        # Integer fields
        "lead_time": (int, float),
        "arrival_date_year": (int, float),
        "arrival_date_week_number": (int, float),
        "arrival_date_day_of_month": (int, float),
        "stays_in_weekend_nights": (int, float),
        "stays_in_week_nights": (int, float),
        "adults": (int, float),
        "children": (int, float),
        "babies": (int, float),
        "is_repeated_guest": (int, float),
        "previous_cancellations": (int, float),
        "previous_bookings_not_canceled": (int, float),
        "booking_changes": (int, float),
        "days_in_waiting_list": (int, float),
        "required_car_parking_spaces": (int, float),
        "total_of_special_requests": (int, float),
        
        # Float fields
        "adr": (int, float),
        "agent": (int, float, type(None)),
        "company": (int, float, type(None))
    }
    
    # Define valid value ranges for numerical fields
    VALUE_RANGES = {
        "lead_time": (0, 1000),  # 0 to ~3 years
        "arrival_date_year": (2015, 2030),  # Reasonable year range
        "arrival_date_week_number": (1, 53),
        "arrival_date_day_of_month": (1, 31),
        "stays_in_weekend_nights": (0, 30),
        "stays_in_week_nights": (0, 50),
        "adults": (0, 10),
        "children": (0, 10),
        "babies": (0, 10),
        "is_repeated_guest": (0, 1),
        "previous_cancellations": (0, 50),
        "previous_bookings_not_canceled": (0, 100),
        "booking_changes": (0, 20),
        "days_in_waiting_list": (0, 500),
        "adr": (0, 1000),  # Average daily rate
        "required_car_parking_spaces": (0, 10),
        "total_of_special_requests": (0, 10)
    }
    
    # Define valid categorical values
    VALID_CATEGORIES = {
        "hotel": ["Resort Hotel", "City Hotel"],
        "meal": ["BB", "HB", "FB", "SC", "Undefined"],
        "deposit_type": ["No Deposit", "Refundable", "Non Refund"],
        "market_segment": [
            "Online TA", "Offline TA/TO", "Groups", "Direct",
            "Corporate", "Complementary", "Aviation", "Undefined"
        ],
        "distribution_channel": [
            "TA/TO", "Direct", "Corporate", "GDS", "Undefined"
        ],
        "customer_type": [
            "Transient", "Contract", "Transient-Party", "Group"
        ],
        "arrival_date_month": [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
    }
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize the InputValidator.
        
        Args:
            strict_mode: If True, enforce strict validation including categorical values.
                        If False, allow unknown categorical values with a warning.
        """
        self.strict_mode = strict_mode
        logger.debug(f"InputValidator initialized (strict_mode={strict_mode})")
    
    def validate_booking_data(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate booking data for prediction.
        
        This is the main validation method that performs all checks:
        - Required fields presence
        - Data types
        - Value ranges
        - Categorical values (if strict_mode)
        
        Args:
            data: Dictionary containing booking data
        
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if validation passes, False otherwise
            - error_message: Empty string if valid, detailed error message if invalid
        """
        if not isinstance(data, dict):
            return False, "Booking data must be a dictionary"
        
        if not data:
            return False, "Booking data is empty"
        
        # Check required fields
        is_valid, error_msg = self.check_required_fields(data)
        if not is_valid:
            return False, error_msg
        
        # Validate data types
        is_valid, error_msg = self.validate_data_types(data)
        if not is_valid:
            return False, error_msg
        
        # Validate value ranges
        is_valid, error_msg = self.validate_value_ranges(data)
        if not is_valid:
            return False, error_msg
        
        # Validate categorical values (if strict mode)
        if self.strict_mode:
            is_valid, error_msg = self.validate_categorical_values(data)
            if not is_valid:
                return False, error_msg
        
        # Additional business logic validations
        is_valid, error_msg = self.validate_business_rules(data)
        if not is_valid:
            return False, error_msg
        
        logger.debug("Booking data validation passed")
        return True, ""
    
    def check_required_fields(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check that all required fields are present in the booking data.
        
        Args:
            data: Dictionary containing booking data
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        missing_fields = []
        
        for field in self.REQUIRED_FIELDS:
            if field not in data:
                missing_fields.append(field)
        
        if missing_fields:
            error_msg = f"Missing required fields: {', '.join(missing_fields)}"
            logger.warning(error_msg)
            return False, error_msg
        
        return True, ""
    
    def validate_data_types(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate that all fields have the correct data types.
        
        Args:
            data: Dictionary containing booking data
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        type_errors = []
        
        for field, value in data.items():
            if field not in self.FIELD_TYPES:
                # Skip fields not in our schema (might be extra fields)
                continue
            
            expected_types = self.FIELD_TYPES[field]
            
            # Handle None values for optional fields
            if value is None:
                if field in self.OPTIONAL_FIELDS:
                    continue
                else:
                    type_errors.append(
                        f"Field '{field}' cannot be None (expected {expected_types})"
                    )
                    continue
            
            # Check if value matches expected type(s)
            if not isinstance(value, expected_types):
                if isinstance(expected_types, tuple):
                    type_names = " or ".join([t.__name__ for t in expected_types if t is not type(None)])
                else:
                    type_names = expected_types.__name__
                
                type_errors.append(
                    f"Field '{field}' has invalid type {type(value).__name__} "
                    f"(expected {type_names})"
                )
        
        if type_errors:
            error_msg = "Data type validation failed: " + "; ".join(type_errors)
            logger.warning(error_msg)
            return False, error_msg
        
        return True, ""
    
    def validate_value_ranges(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate that numerical values are within acceptable ranges.
        
        Args:
            data: Dictionary containing booking data
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        range_errors = []
        
        for field, (min_val, max_val) in self.VALUE_RANGES.items():
            if field not in data:
                continue
            
            value = data[field]
            
            # Skip None values for optional fields
            if value is None and field in self.OPTIONAL_FIELDS:
                continue
            
            # Check if value is within range
            try:
                numeric_value = float(value)
                
                if numeric_value < min_val or numeric_value > max_val:
                    range_errors.append(
                        f"Field '{field}' value {numeric_value} is out of range "
                        f"(expected {min_val} to {max_val})"
                    )
            except (ValueError, TypeError):
                # This should be caught by type validation, but handle it anyway
                range_errors.append(
                    f"Field '{field}' value '{value}' cannot be converted to number"
                )
        
        if range_errors:
            error_msg = "Value range validation failed: " + "; ".join(range_errors)
            logger.warning(error_msg)
            return False, error_msg
        
        return True, ""
    
    def validate_categorical_values(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate that categorical fields have valid values.
        
        This method is only used in strict mode. In non-strict mode,
        unknown categorical values are allowed with a warning.
        
        Args:
            data: Dictionary containing booking data
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        category_errors = []
        
        for field, valid_values in self.VALID_CATEGORIES.items():
            if field not in data:
                continue
            
            value = data[field]
            
            if value not in valid_values:
                category_errors.append(
                    f"Field '{field}' has invalid value '{value}' "
                    f"(expected one of: {', '.join(valid_values)})"
                )
        
        if category_errors:
            error_msg = "Categorical validation failed: " + "; ".join(category_errors)
            logger.warning(error_msg)
            return False, error_msg
        
        return True, ""
    
    def validate_business_rules(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate business logic rules for booking data.
        
        These are domain-specific rules that ensure data makes logical sense,
        such as:
        - Total guests (adults + children + babies) must be > 0
        - Total nights (weekend + week nights) must be > 0
        - ADR (average daily rate) should be > 0 for paid bookings
        
        Args:
            data: Dictionary containing booking data
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        business_errors = []
        
        # Rule 1: Total guests must be greater than 0
        adults = data.get("adults", 0)
        children = data.get("children", 0)
        babies = data.get("babies", 0)
        total_guests = adults + children + babies
        
        if total_guests <= 0:
            business_errors.append(
                f"Total guests (adults + children + babies) must be greater than 0 "
                f"(got {total_guests})"
            )
        
        # Rule 2: Total nights must be greater than 0
        weekend_nights = data.get("stays_in_weekend_nights", 0)
        week_nights = data.get("stays_in_week_nights", 0)
        total_nights = weekend_nights + week_nights
        
        if total_nights <= 0:
            business_errors.append(
                f"Total nights (weekend + week nights) must be greater than 0 "
                f"(got {total_nights})"
            )
        
        # Rule 3: ADR should be positive for most bookings
        adr = data.get("adr", 0)
        if adr < 0:
            business_errors.append(
                f"Average daily rate (adr) cannot be negative (got {adr})"
            )
        
        # Rule 4: Lead time should be non-negative
        lead_time = data.get("lead_time", 0)
        if lead_time < 0:
            business_errors.append(
                f"Lead time cannot be negative (got {lead_time})"
            )
        
        # Rule 5: Previous cancellations and bookings should be consistent
        prev_cancellations = data.get("previous_cancellations", 0)
        prev_bookings = data.get("previous_bookings_not_canceled", 0)
        is_repeated = data.get("is_repeated_guest", 0)
        
        if is_repeated == 1 and (prev_cancellations + prev_bookings) == 0:
            # Warning only, not a hard error
            logger.warning(
                "Booking marked as repeated guest but has no previous booking history"
            )
        
        if business_errors:
            error_msg = "Business rule validation failed: " + "; ".join(business_errors)
            logger.warning(error_msg)
            return False, error_msg
        
        return True, ""
    
    def get_required_fields(self) -> List[str]:
        """
        Get the list of required fields for booking data.
        
        Returns:
            List of required field names
        """
        return self.REQUIRED_FIELDS.copy()
    
    def get_optional_fields(self) -> List[str]:
        """
        Get the list of optional fields for booking data.
        
        Returns:
            List of optional field names
        """
        return self.OPTIONAL_FIELDS.copy()
    
    def get_field_info(self, field_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific field.
        
        Args:
            field_name: Name of the field
        
        Returns:
            Dictionary with field information (type, range, valid values)
            or None if field not found
        """
        if field_name not in self.FIELD_TYPES:
            return None
        
        info = {
            "name": field_name,
            "required": field_name in self.REQUIRED_FIELDS,
            "type": self.FIELD_TYPES[field_name]
        }
        
        if field_name in self.VALUE_RANGES:
            min_val, max_val = self.VALUE_RANGES[field_name]
            info["min_value"] = min_val
            info["max_value"] = max_val
        
        if field_name in self.VALID_CATEGORIES:
            info["valid_values"] = self.VALID_CATEGORIES[field_name]
        
        return info
