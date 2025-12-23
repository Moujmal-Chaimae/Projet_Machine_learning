"""
Example usage of visualization components for the Hotel Cancellation Predictor.

This script demonstrates how to use the visualization functions to create
interactive charts for displaying prediction results.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.components.visualizations import (
    plot_probability_gauge,
    plot_feature_importance,
    plot_risk_distribution,
    plot_probability_distribution,
    plot_feature_contributions,
    create_summary_metrics_chart
)


def example_probability_gauge():
    """Example: Create a probability gauge chart."""
    print("Creating probability gauge chart...")
    
    # Example probability (75% chance of cancellation)
    probability = 0.75
    
    fig = plot_probability_gauge(probability)
    
    # Save to HTML file
    fig.write_html("examples/output/probability_gauge.html")
    print("✓ Saved to examples/output/probability_gauge.html")
    
    # Or display in browser
    # fig.show()


def example_feature_importance():
    """Example: Create a feature importance bar chart."""
    print("\nCreating feature importance chart...")
    
    # Example feature importance scores
    feature_importance = {
        'lead_time': 0.2543,
        'adr': 0.1876,
        'deposit_type': 0.1432,
        'total_of_special_requests': 0.0987,
        'required_car_parking_spaces': 0.0854,
        'previous_cancellations': 0.0765,
        'booking_changes': 0.0654,
        'days_in_waiting_list': 0.0543,
        'customer_type': 0.0432,
        'market_segment': 0.0321,
        'country': 0.0287,
        'is_repeated_guest': 0.0198
    }
    
    fig = plot_feature_importance(feature_importance, top_n=10)
    
    # Save to HTML file
    fig.write_html("examples/output/feature_importance.html")
    print("✓ Saved to examples/output/feature_importance.html")


def example_risk_distribution():
    """Example: Create a risk distribution pie chart."""
    print("\nCreating risk distribution chart...")
    
    # Example predictions with different risk levels
    predictions = [
        {'risk_level': 'low', 'probability': 0.15},
        {'risk_level': 'low', 'probability': 0.22},
        {'risk_level': 'low', 'probability': 0.28},
        {'risk_level': 'medium', 'probability': 0.45},
        {'risk_level': 'medium', 'probability': 0.55},
        {'risk_level': 'medium', 'probability': 0.62},
        {'risk_level': 'high', 'probability': 0.78},
        {'risk_level': 'high', 'probability': 0.85},
        {'risk_level': 'high', 'probability': 0.92},
        {'risk_level': 'high', 'probability': 0.88}
    ]
    
    fig = plot_risk_distribution(predictions)
    
    # Save to HTML file
    fig.write_html("examples/output/risk_distribution.html")
    print("✓ Saved to examples/output/risk_distribution.html")


def example_probability_distribution():
    """Example: Create a probability distribution histogram."""
    print("\nCreating probability distribution histogram...")
    
    # Example predictions with various probabilities
    import random
    random.seed(42)
    
    predictions = [
        {'probability': random.random()}
        for _ in range(100)
    ]
    
    fig = plot_probability_distribution(predictions, bins=20)
    
    # Save to HTML file
    fig.write_html("examples/output/probability_distribution.html")
    print("✓ Saved to examples/output/probability_distribution.html")


def example_feature_contributions():
    """Example: Create a feature contributions chart for a specific booking."""
    print("\nCreating feature contributions chart...")
    
    # Example booking data
    booking_data = {
        'lead_time': 342,
        'adr': 95.5,
        'deposit_type': 'No Deposit',
        'total_of_special_requests': 2,
        'required_car_parking_spaces': 0,
        'previous_cancellations': 0,
        'booking_changes': 1,
        'days_in_waiting_list': 0,
        'customer_type': 'Transient',
        'market_segment': 'Online TA'
    }
    
    # Example feature importance
    feature_importance = {
        'lead_time': 0.2543,
        'adr': 0.1876,
        'deposit_type': 0.1432,
        'total_of_special_requests': 0.0987,
        'required_car_parking_spaces': 0.0854,
        'previous_cancellations': 0.0765,
        'booking_changes': 0.0654,
        'days_in_waiting_list': 0.0543,
        'customer_type': 0.0432,
        'market_segment': 0.0321
    }
    
    fig = plot_feature_contributions(booking_data, feature_importance, top_n=10)
    
    # Save to HTML file
    fig.write_html("examples/output/feature_contributions.html")
    print("✓ Saved to examples/output/feature_contributions.html")


def example_summary_metrics():
    """Example: Create a summary metrics chart for batch predictions."""
    print("\nCreating summary metrics chart...")
    
    # Example batch predictions
    import random
    random.seed(42)
    
    predictions = []
    for i in range(50):
        prob = random.random()
        predictions.append({
            'prediction': 1 if prob > 0.5 else 0,
            'probability': prob,
            'risk_level': 'high' if prob > 0.7 else ('medium' if prob > 0.3 else 'low')
        })
    
    fig = create_summary_metrics_chart(predictions)
    
    # Save to HTML file
    fig.write_html("examples/output/summary_metrics.html")
    print("✓ Saved to examples/output/summary_metrics.html")


def main():
    """Run all visualization examples."""
    print("=" * 60)
    print("Hotel Cancellation Predictor - Visualization Examples")
    print("=" * 60)
    
    # Create output directory if it doesn't exist
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True)
    
    # Run all examples
    example_probability_gauge()
    example_feature_importance()
    example_risk_distribution()
    example_probability_distribution()
    example_feature_contributions()
    example_summary_metrics()
    
    print("\n" + "=" * 60)
    print("✓ All visualizations created successfully!")
    print("✓ Check the 'examples/output/' directory for HTML files")
    print("=" * 60)


if __name__ == "__main__":
    main()
