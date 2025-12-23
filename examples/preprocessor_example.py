"""
Example demonstrating the Preprocessor usage for new predictions.

This script shows how to:
1. Load fitted transformers
2. Preprocess new booking data
3. Use the preprocessor with the prediction service
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.prediction.preprocessor import Preprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def example_basic_preprocessing():
    """Example of basic preprocessing with the Preprocessor."""
    
    print("\n" + "="*70)
    print("Example 1: Basic Preprocessing")
    print("="*70)
    
    # Sample booking data
    booking_data = {
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
    
    # Configuration (should match training configuration)
    label_encode_cols = ["hotel", "meal", "deposit_type"]
    onehot_encode_cols = ["market_segment", "distribution_channel", "customer_type"]
    features_to_drop = ["reservation_status", "reservation_status_date", "agent", "company"]
    
    # Initialize preprocessor
    preprocessor = Preprocessor(
        label_encode_cols=label_encode_cols,
        onehot_encode_cols=onehot_encode_cols,
        features_to_drop=features_to_drop
    )
    
    # Note: In real usage, you would load fitted transformers:
    # preprocessor.load_transformers("data/processed/")
    
    print("\nBooking data:")
    print(f"  Hotel: {booking_data['hotel']}")
    print(f"  Lead time: {booking_data['lead_time']} days")
    print(f"  Adults: {booking_data['adults']}, Children: {booking_data['children']}")
    print(f"  Total nights: {booking_data['stays_in_weekend_nights'] + booking_data['stays_in_week_nights']}")
    print(f"  ADR: ${booking_data['adr']}")
    
    print("\nPreprocessor configuration:")
    print(f"  Label encode: {label_encode_cols}")
    print(f"  One-hot encode: {onehot_encode_cols}")
    print(f"  Features to drop: {features_to_drop}")
    
    # Note: Actual transformation requires fitted transformers
    print("\nNote: To use transform(), you must first load fitted transformers")
    print("      from the training phase using load_transformers()")


def example_with_multiple_bookings():
    """Example of preprocessing multiple bookings at once."""
    
    print("\n" + "="*70)
    print("Example 2: Batch Preprocessing")
    print("="*70)
    
    # Multiple bookings
    bookings = [
        {
            "hotel": "Resort Hotel",
            "lead_time": 120,
            "adults": 2,
            "children": 1,
            "babies": 0,
            "stays_in_weekend_nights": 2,
            "stays_in_week_nights": 3,
            "adr": 75.5,
            # ... other fields ...
        },
        {
            "hotel": "City Hotel",
            "lead_time": 30,
            "adults": 1,
            "children": 0,
            "babies": 0,
            "stays_in_weekend_nights": 0,
            "stays_in_week_nights": 2,
            "adr": 95.0,
            # ... other fields ...
        }
    ]
    
    print(f"\nProcessing {len(bookings)} bookings:")
    for i, booking in enumerate(bookings, 1):
        print(f"  Booking {i}: {booking['hotel']}, {booking['adults']} adults, "
              f"{booking['lead_time']} days lead time")
    
    print("\nThe Preprocessor can handle:")
    print("  - Single booking (dict)")
    print("  - Multiple bookings (list of dicts)")
    print("  - DataFrame with multiple rows")


def example_feature_engineering():
    """Example showing what features are created during preprocessing."""
    
    print("\n" + "="*70)
    print("Example 3: Feature Engineering")
    print("="*70)
    
    print("\nDerived features created by the preprocessor:")
    print("  1. total_guests = adults + children + babies")
    print("  2. total_nights = weekend_nights + week_nights")
    print("  3. has_children = 1 if (children > 0 or babies > 0) else 0")
    print("  4. is_long_stay = 1 if total_nights > 7 else 0")
    print("  5. price_per_night_per_guest = adr / (total_guests * total_nights)")
    print("  6. room_type_match = 1 if reserved == assigned else 0")
    print("  7. has_special_requests = 1 if total_special_requests > 0 else 0")
    
    print("\nLog transformations applied to skewed features:")
    print("  - Features with skewness > 1.0 get log transformation")
    print("  - Example: lead_time_log = log(1 + lead_time)")
    
    print("\nEncoding transformations:")
    print("  - Label encoding: hotel, meal, deposit_type")
    print("  - One-hot encoding: market_segment, distribution_channel, customer_type")
    
    print("\nScaling transformations:")
    print("  - StandardScaler (z-score normalization) for numerical features")
    print("  - Binary features (0/1) are not scaled")


def example_integration_with_prediction_service():
    """Example showing how Preprocessor integrates with PredictionService."""
    
    print("\n" + "="*70)
    print("Example 4: Integration with Prediction Service")
    print("="*70)
    
    print("\nTypical workflow:")
    print("  1. Train model and save fitted transformers")
    print("     - feature_engineer.save_transformers('data/processed/')")
    print()
    print("  2. Initialize Preprocessor with saved transformers")
    print("     - preprocessor = Preprocessor(transformers_path='data/processed/')")
    print()
    print("  3. Use Preprocessor in PredictionService")
    print("     - service = PredictionService(")
    print("         model_path='models/best_model.pkl',")
    print("         preprocessor=preprocessor")
    print("       )")
    print()
    print("  4. Make predictions")
    print("     - result = service.predict(booking_data)")
    
    print("\nThe Preprocessor ensures:")
    print("  ✓ Same transformations as training")
    print("  ✓ Consistent feature engineering")
    print("  ✓ Proper encoding and scaling")
    print("  ✓ Correct feature order for model input")


def main():
    """Run all examples."""
    
    print("\n" + "="*70)
    print("PREPROCESSOR EXAMPLES")
    print("="*70)
    print("\nThis script demonstrates how to use the Preprocessor class")
    print("for preprocessing new booking data before making predictions.")
    
    try:
        example_basic_preprocessing()
        example_with_multiple_bookings()
        example_feature_engineering()
        example_integration_with_prediction_service()
        
        print("\n" + "="*70)
        print("Examples completed successfully!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Train a model and save transformers")
        print("  2. Load transformers in Preprocessor")
        print("  3. Use Preprocessor to transform new booking data")
        print("  4. Make predictions with the transformed data")
        print()
        
    except Exception as e:
        logger.error(f"Example failed: {str(e)}")
        print(f"\nError: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
