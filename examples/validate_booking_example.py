"""
Example demonstrating the InputValidator usage.

This script shows how to validate booking data before making predictions.
"""

from src.prediction.input_validator import InputValidator


def main():
    """Demonstrate input validation."""
    
    # Create validator
    validator = InputValidator(strict_mode=False)
    
    print("=" * 60)
    print("Hotel Booking Input Validation Example")
    print("=" * 60)
    
    # Example 1: Valid booking data
    print("\n1. Testing VALID booking data:")
    valid_booking = {
        "hotel": "Resort Hotel",
        "lead_time": 120,
        "arrival_date_year": 2024,
        "arrival_date_month": "July",
        "arrival_date_week_number": 28,
        "arrival_date_day_of_month": 15,
        "stays_in_weekend_nights": 2,
        "stays_in_week_nights": 3,
        "adults": 2,
        "children": 1,
        "babies": 0,
        "meal": "BB",
        "country": "PRT",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A",
        "assigned_room_type": "A",
        "booking_changes": 0,
        "deposit_type": "No Deposit",
        "agent": 240.0,
        "company": None,
        "days_in_waiting_list": 0,
        "customer_type": "Transient",
        "adr": 75.5,
        "required_car_parking_spaces": 0,
        "total_of_special_requests": 1
    }
    
    is_valid, error_msg = validator.validate_booking_data(valid_booking)
    print(f"   Valid: {is_valid}")
    if error_msg:
        print(f"   Error: {error_msg}")
    else:
        print("   ✓ All validation checks passed!")
    
    # Example 2: Missing required field
    print("\n2. Testing booking with MISSING FIELD:")
    invalid_booking = valid_booking.copy()
    del invalid_booking["hotel"]
    
    is_valid, error_msg = validator.validate_booking_data(invalid_booking)
    print(f"   Valid: {is_valid}")
    print(f"   Error: {error_msg}")
    
    # Example 3: Invalid data type
    print("\n3. Testing booking with INVALID DATA TYPE:")
    invalid_booking = valid_booking.copy()
    invalid_booking["lead_time"] = "not a number"
    
    is_valid, error_msg = validator.validate_booking_data(invalid_booking)
    print(f"   Valid: {is_valid}")
    print(f"   Error: {error_msg}")
    
    # Example 4: Value out of range
    print("\n4. Testing booking with VALUE OUT OF RANGE:")
    invalid_booking = valid_booking.copy()
    invalid_booking["adults"] = 50  # Max is 10
    
    is_valid, error_msg = validator.validate_booking_data(invalid_booking)
    print(f"   Valid: {is_valid}")
    print(f"   Error: {error_msg}")
    
    # Example 5: Business rule violation (zero guests)
    print("\n5. Testing booking with BUSINESS RULE VIOLATION:")
    invalid_booking = valid_booking.copy()
    invalid_booking["adults"] = 0
    invalid_booking["children"] = 0
    invalid_booking["babies"] = 0
    
    is_valid, error_msg = validator.validate_booking_data(invalid_booking)
    print(f"   Valid: {is_valid}")
    print(f"   Error: {error_msg}")
    
    # Show field information
    print("\n6. Getting FIELD INFORMATION:")
    field_info = validator.get_field_info("lead_time")
    print(f"   Field: {field_info['name']}")
    print(f"   Required: {field_info['required']}")
    print(f"   Type: {field_info['type']}")
    print(f"   Range: {field_info['min_value']} to {field_info['max_value']}")
    
    print("\n" + "=" * 60)
    print("Validation examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
