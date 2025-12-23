"""
Prediction Display Component for Hotel Cancellation Prediction Web Interface

This module provides functions to display prediction results in a user-friendly
format with visual indicators for risk levels and cancellation probabilities.
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any


def display_prediction_result(result: Dict[str, Any]) -> None:
    """
    Display the prediction result with visual indicators and detailed information.
    
    This function renders a comprehensive display of the cancellation prediction,
    including binary prediction, probability, risk level, and timestamp.
    
    Args:
        result: Dictionary containing prediction results with keys:
            - prediction: Binary prediction (0 or 1)
            - probability: Cancellation probability (0.0 to 1.0)
            - risk_level: Risk category ('low', 'medium', 'high')
            - confidence: Model confidence score (optional)
            - timestamp: Prediction timestamp (ISO format)
            - prediction_time_ms: Prediction time in milliseconds (optional)
    
    Example:
        >>> result = {
        ...     "prediction": 1,
        ...     "probability": 0.75,
        ...     "risk_level": "high",
        ...     "confidence": 0.85,
        ...     "timestamp": "2025-12-17T10:30:00"
        ... }
        >>> display_prediction_result(result)
    """
    
    # Extract prediction data
    prediction = result.get('prediction', 0)
    probability = result.get('probability', 0.0)
    risk_level = result.get('risk_level', 'unknown').lower()
    confidence = result.get('confidence', 0.0)
    timestamp = result.get('timestamp', datetime.now().isoformat())
    prediction_time_ms = result.get('prediction_time_ms', 0)
    
    # Display main prediction header
    st.markdown("## 🎯 Prediction Results")
    st.markdown("---")
    
    # Binary prediction with colored badge
    st.markdown("### Cancellation Prediction")
    
    if prediction == 1:
        # Will Cancel - Red badge
        st.markdown(
            """
            <div style="
                background-color: #ff4b4b;
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            ">
                ❌ WILL CANCEL
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Will Not Cancel - Green badge
        st.markdown(
            """
            <div style="
                background-color: #00cc66;
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            ">
                ✅ WILL NOT CANCEL
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Cancellation probability with progress bar
    st.markdown("### 📊 Cancellation Probability")
    
    probability_percent = probability * 100
    
    # Display probability as percentage
    st.markdown(
        f"""
        <div style="
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            color: #1f77b4;
            margin: 20px 0;
        ">
            {probability_percent:.1f}%
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Progress bar with color based on probability
    if probability < 0.3:
        bar_color = "#00cc66"  # Green for low risk
    elif probability < 0.7:
        bar_color = "#ffa500"  # Orange for medium risk
    else:
        bar_color = "#ff4b4b"  # Red for high risk
    
    st.progress(probability, text=f"Cancellation Likelihood: {probability_percent:.1f}%")
    
    st.markdown("---")
    
    # Risk level with color coding
    st.markdown("### ⚠️ Risk Level")
    
    risk_colors = {
        "low": ("#00cc66", "🟢", "Low Risk"),
        "medium": ("#ffa500", "🟡", "Medium Risk"),
        "high": ("#ff4b4b", "🔴", "High Risk")
    }
    
    color, emoji, label = risk_colors.get(risk_level, ("#808080", "⚪", "Unknown Risk"))
    
    st.markdown(
        f"""
        <div style="
            background-color: {color};
            color: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            margin: 10px 0;
        ">
            {emoji} {label.upper()}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Risk level interpretation
    if risk_level == "low":
        st.success(
            "**Low Risk (< 30%):** This booking has a low probability of cancellation. "
            "The guest is likely to honor their reservation."
        )
    elif risk_level == "medium":
        st.warning(
            "**Medium Risk (30-70%):** This booking has a moderate probability of cancellation. "
            "Consider monitoring this reservation or implementing flexible policies."
        )
    elif risk_level == "high":
        st.error(
            "**High Risk (> 70%):** This booking has a high probability of cancellation. "
            "Consider overbooking strategies or contacting the guest to confirm."
        )
    
    st.markdown("---")
    
    # Additional details section
    st.markdown("### 📋 Additional Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Confidence Score",
            value=f"{confidence * 100:.1f}%",
            help="Model confidence in this prediction"
        )
        
        if prediction_time_ms > 0:
            st.metric(
                label="Prediction Time",
                value=f"{prediction_time_ms:.2f} ms",
                help="Time taken to generate this prediction"
            )
    
    with col2:
        # Format timestamp for display
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            formatted_time = timestamp
        
        st.metric(
            label="Prediction Timestamp",
            value=formatted_time,
            help="When this prediction was generated"
        )
        
        st.metric(
            label="Binary Prediction",
            value=prediction,
            help="0 = No Cancellation, 1 = Cancellation"
        )


def display_risk_summary(probability: float) -> None:
    """
    Display a summary of risk categories and where the current prediction falls.
    
    Args:
        probability: Cancellation probability (0.0 to 1.0)
    """
    st.markdown("### 📈 Risk Category Breakdown")
    
    # Create visual representation of risk zones
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div style="
                background-color: #00cc66;
                color: white;
                padding: 10px;
                border-radius: 5px;
                text-align: center;
            ">
                🟢 Low Risk<br>0% - 30%
            </div>
            """,
            unsafe_allow_html=True
        )
        if probability < 0.3:
            st.markdown("**← Current Prediction**")
    
    with col2:
        st.markdown(
            """
            <div style="
                background-color: #ffa500;
                color: white;
                padding: 10px;
                border-radius: 5px;
                text-align: center;
            ">
                🟡 Medium Risk<br>30% - 70%
            </div>
            """,
            unsafe_allow_html=True
        )
        if 0.3 <= probability < 0.7:
            st.markdown("**← Current Prediction**")
    
    with col3:
        st.markdown(
            """
            <div style="
                background-color: #ff4b4b;
                color: white;
                padding: 10px;
                border-radius: 5px;
                text-align: center;
            ">
                🔴 High Risk<br>70% - 100%
            </div>
            """,
            unsafe_allow_html=True
        )
        if probability >= 0.7:
            st.markdown("**← Current Prediction**")


def display_compact_prediction(result: Dict[str, Any]) -> None:
    """
    Display a compact version of the prediction result.
    
    Useful for batch predictions or when space is limited.
    
    Args:
        result: Dictionary containing prediction results
    """
    prediction = result.get('prediction', 0)
    probability = result.get('probability', 0.0)
    risk_level = result.get('risk_level', 'unknown').lower()
    
    # Risk level emoji mapping
    risk_emoji = {
        "low": "🟢",
        "medium": "🟡",
        "high": "🔴"
    }
    
    emoji = risk_emoji.get(risk_level, "⚪")
    
    # Prediction label
    prediction_label = "WILL CANCEL" if prediction == 1 else "WON'T CANCEL"
    
    # Display in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Prediction", prediction_label)
    
    with col2:
        st.metric("Probability", f"{probability * 100:.1f}%")
    
    with col3:
        st.metric("Risk", f"{emoji} {risk_level.upper()}")


def display_prediction_explanation(result: Dict[str, Any]) -> None:
    """
    Display an explanation of what the prediction means and recommended actions.
    
    Args:
        result: Dictionary containing prediction results
    """
    prediction = result.get('prediction', 0)
    probability = result.get('probability', 0.0)
    risk_level = result.get('risk_level', 'unknown').lower()
    
    st.markdown("### 💡 What This Means")
    
    if prediction == 1:
        st.markdown("""
        **The model predicts this booking will be cancelled.**
        
        Based on the booking characteristics, our machine learning model has identified
        patterns that are commonly associated with cancellations.
        """)
    else:
        st.markdown("""
        **The model predicts this booking will NOT be cancelled.**
        
        Based on the booking characteristics, our machine learning model has identified
        patterns that are commonly associated with confirmed reservations.
        """)
    
    st.markdown("### 🎯 Recommended Actions")
    
    if risk_level == "low":
        st.info("""
        **Low Risk Recommendations:**
        - Proceed with standard booking procedures
        - No special monitoring required
        - Guest is likely to honor the reservation
        """)
    elif risk_level == "medium":
        st.warning("""
        **Medium Risk Recommendations:**
        - Monitor this booking more closely
        - Consider sending a confirmation reminder
        - Have backup bookings ready if possible
        - Review cancellation policy with guest
        """)
    else:  # high risk
        st.error("""
        **High Risk Recommendations:**
        - Contact guest to confirm reservation
        - Consider requiring a deposit or prepayment
        - Implement overbooking strategy for this room
        - Prepare alternative booking options
        - Review and enforce strict cancellation policy
        """)
