"""
Example usage of the input form component.

This script demonstrates how to use the render_input_form() function
in a Streamlit application.

To run this example:
    streamlit run examples/input_form_example.py
"""

import streamlit as st
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.components.input_form import render_input_form, render_quick_input_form


def main():
    """Main function to demonstrate input form usage."""
    
    st.set_page_config(
        page_title="Input Form Example",
        page_icon="🏨",
        layout="wide"
    )
    
    st.title("🏨 Hotel Cancellation Prediction - Input Form Example")
    st.markdown("---")
    
    # Sidebar for form selection
    st.sidebar.title("Form Options")
    form_type = st.sidebar.radio(
        "Select Form Type",
        ["Full Form", "Quick Form"],
        help="Choose between detailed or quick input form"
    )
    
    # Render selected form
    if form_type == "Full Form":
        st.info("📋 This is the full detailed form with all booking features.")
        booking_data = render_input_form()
    else:
        st.info("⚡ This is the quick form with only essential features.")
        booking_data = render_quick_input_form()
    
    # Display submitted data
    if booking_data:
        st.success("✅ Form submitted successfully!")
        
        st.markdown("### 📊 Submitted Booking Data")
        
        # Display in columns for better readability
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Basic Information")
            st.json({
                "hotel": booking_data["hotel"],
                "lead_time": booking_data["lead_time"],
                "adr": booking_data["adr"],
                "arrival_date_year": booking_data["arrival_date_year"],
                "arrival_date_month": booking_data["arrival_date_month"]
            })
            
            st.markdown("#### Guest Information")
            st.json({
                "adults": booking_data["adults"],
                "children": booking_data["children"],
                "babies": booking_data["babies"],
                "total_guests": booking_data["adults"] + booking_data["children"] + booking_data["babies"]
            })
        
        with col2:
            st.markdown("#### Stay Details")
            st.json({
                "stays_in_weekend_nights": booking_data["stays_in_weekend_nights"],
                "stays_in_week_nights": booking_data["stays_in_week_nights"],
                "total_nights": booking_data["stays_in_weekend_nights"] + booking_data["stays_in_week_nights"],
                "meal": booking_data["meal"]
            })
            
            st.markdown("#### Booking Details")
            st.json({
                "deposit_type": booking_data["deposit_type"],
                "customer_type": booking_data["customer_type"],
                "market_segment": booking_data["market_segment"],
                "previous_cancellations": booking_data["previous_cancellations"]
            })
        
        # Display all data in expandable section
        with st.expander("🔍 View All Submitted Data"):
            st.json(booking_data)
        
        # Simulate prediction (placeholder)
        st.markdown("---")
        st.markdown("### 🔮 Prediction Result (Placeholder)")
        st.info("In a real application, this data would be sent to the PredictionService for processing.")
        
        # Show what would happen next
        st.code("""
# Example of how this data would be used:
from src.prediction.prediction_service import PredictionService

# Initialize prediction service
service = PredictionService(
    model_path='models/best_model.pkl',
    preprocessor_path='models/preprocessor.pkl'
)

# Make prediction
result = service.predict(booking_data)

# Display result
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
        """, language="python")


if __name__ == "__main__":
    main()
