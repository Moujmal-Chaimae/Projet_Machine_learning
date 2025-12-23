"""
Example usage of the prediction display component.

This script demonstrates how to use the display_prediction_result function
to show prediction results in a Streamlit application.

To run this example:
    streamlit run examples/prediction_display_example.py
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.components.prediction_display import (
    display_prediction_result,
    display_risk_summary,
    display_compact_prediction,
    display_prediction_explanation
)


def main():
    st.set_page_config(
        page_title="Prediction Display Example",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 Prediction Display Component Example")
    st.markdown("This example demonstrates the prediction display component.")
    
    # Sidebar for selecting example scenarios
    st.sidebar.title("Example Scenarios")
    
    scenario = st.sidebar.selectbox(
        "Select a scenario:",
        [
            "High Risk Cancellation",
            "Low Risk Booking",
            "Medium Risk Booking",
            "Custom Probability"
        ]
    )
    
    # Define example results based on scenario
    if scenario == "High Risk Cancellation":
        result = {
            "prediction": 1,
            "probability": 0.85,
            "risk_level": "high",
            "confidence": 0.70,
            "timestamp": datetime.now().isoformat(),
            "prediction_time_ms": 45.23
        }
    elif scenario == "Low Risk Booking":
        result = {
            "prediction": 0,
            "probability": 0.15,
            "risk_level": "low",
            "confidence": 0.70,
            "timestamp": datetime.now().isoformat(),
            "prediction_time_ms": 38.67
        }
    elif scenario == "Medium Risk Booking":
        result = {
            "prediction": 1,
            "probability": 0.55,
            "risk_level": "medium",
            "confidence": 0.10,
            "timestamp": datetime.now().isoformat(),
            "prediction_time_ms": 42.15
        }
    else:  # Custom Probability
        custom_prob = st.sidebar.slider(
            "Set custom probability:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01
        )
        
        # Determine risk level and prediction
        if custom_prob < 0.3:
            risk = "low"
        elif custom_prob < 0.7:
            risk = "medium"
        else:
            risk = "high"
        
        pred = 1 if custom_prob >= 0.5 else 0
        
        result = {
            "prediction": pred,
            "probability": custom_prob,
            "risk_level": risk,
            "confidence": abs(custom_prob - 0.5) * 2,
            "timestamp": datetime.now().isoformat(),
            "prediction_time_ms": 40.0
        }
    
    # Display options
    st.sidebar.markdown("---")
    st.sidebar.subheader("Display Options")
    
    show_full = st.sidebar.checkbox("Show Full Display", value=True)
    show_compact = st.sidebar.checkbox("Show Compact Display", value=False)
    show_risk_summary = st.sidebar.checkbox("Show Risk Summary", value=False)
    show_explanation = st.sidebar.checkbox("Show Explanation", value=False)
    
    # Display the selected components
    if show_full:
        st.markdown("## Full Display")
        display_prediction_result(result)
        st.markdown("---")
    
    if show_compact:
        st.markdown("## Compact Display")
        display_compact_prediction(result)
        st.markdown("---")
    
    if show_risk_summary:
        st.markdown("## Risk Summary")
        display_risk_summary(result['probability'])
        st.markdown("---")
    
    if show_explanation:
        st.markdown("## Prediction Explanation")
        display_prediction_explanation(result)
        st.markdown("---")
    
    # Show the result data
    with st.expander("📊 View Result Data"):
        st.json(result)


if __name__ == "__main__":
    main()
