"""
Input Form Component for Hotel Cancellation Prediction Web Interface

This module provides a Streamlit-based input form for collecting booking details
from users to predict cancellation probability.
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, Any


def render_input_form() -> Dict[str, Any]:
    """
    Render an interactive input form for hotel booking details.
    
    This function creates a comprehensive form with appropriate Streamlit widgets
    for all required booking features. Default values and ranges are based on
    data exploration insights from the hotel bookings dataset.
    
    Returns:
        dict: Dictionary containing all user inputs with feature names as keys.
              Returns None if form is not submitted.
    
    Example:
        >>> booking_data = render_input_form()
        >>> if booking_data:
        >>>     # Process the booking data
        >>>     prediction = predict(booking_data)
    """
    
    st.subheader("📝 Enter Booking Details")
    st.markdown("Fill in the booking information below to predict cancellation probability.")
    
    # Create form
    with st.form(key="booking_form"):
        
        # Section 1: Basic Booking Information
        st.markdown("### 🏨 Basic Information")
        col1, col2 = st.columns(2)
        
        with col1:
            hotel = st.selectbox(
                "Hotel Type",
                options=["Resort Hotel", "City Hotel"],
                index=1,
                help="Type of hotel"
            )
            
            lead_time = st.number_input(
                "Lead Time (days)",
                min_value=0,
                max_value=737,
                value=100,
                step=1,
                help="Number of days between booking date and arrival date"
            )
            
            arrival_date_month = st.selectbox(
                "Arrival Month",
                options=["January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December"],
                index=6,
                help="Month of arrival"
            )
        
        with col2:
            adr = st.number_input(
                "Average Daily Rate (ADR)",
                min_value=0.0,
                max_value=5400.0,
                value=100.0,
                step=5.0,
                help="Average daily rate in the currency of the hotel"
            )
            
            arrival_date_year = st.number_input(
                "Arrival Year",
                min_value=2015,
                max_value=2025,
                value=datetime.now().year,
                step=1,
                help="Year of arrival"
            )
            
            arrival_date_week_number = st.number_input(
                "Arrival Week Number",
                min_value=1,
                max_value=53,
                value=27,
                step=1,
                help="Week number of the year for arrival"
            )
        
        # Section 2: Stay Details
        st.markdown("### 🛏️ Stay Details")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            stays_in_weekend_nights = st.number_input(
                "Weekend Nights",
                min_value=0,
                max_value=19,
                value=1,
                step=1,
                help="Number of weekend nights (Saturday or Sunday)"
            )
        
        with col4:
            stays_in_week_nights = st.number_input(
                "Week Nights",
                min_value=0,
                max_value=50,
                value=2,
                step=1,
                help="Number of week nights (Monday to Friday)"
            )
        
        with col5:
            meal = st.selectbox(
                "Meal Plan",
                options=["BB", "HB", "FB", "SC", "Undefined"],
                index=0,
                help="BB: Bed & Breakfast, HB: Half Board, FB: Full Board, SC: Self Catering"
            )
        
        # Section 3: Guest Information
        st.markdown("### 👥 Guest Information")
        col6, col7, col8 = st.columns(3)
        
        with col6:
            adults = st.number_input(
                "Number of Adults",
                min_value=0,
                max_value=55,
                value=2,
                step=1,
                help="Number of adults"
            )
        
        with col7:
            children = st.number_input(
                "Number of Children",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                help="Number of children"
            )
        
        with col8:
            babies = st.number_input(
                "Number of Babies",
                min_value=0,
                max_value=10,
                value=0,
                step=1,
                help="Number of babies"
            )
        
        # Section 4: Booking Channel & Customer Type
        st.markdown("### 📊 Booking Details")
        col9, col10 = st.columns(2)
        
        with col9:
            market_segment = st.selectbox(
                "Market Segment",
                options=["Online TA", "Offline TA/TO", "Direct", "Corporate", 
                        "Groups", "Complementary", "Aviation"],
                index=0,
                help="Market segment designation"
            )
            
            distribution_channel = st.selectbox(
                "Distribution Channel",
                options=["TA/TO", "Direct", "Corporate", "GDS", "Undefined"],
                index=0,
                help="Booking distribution channel"
            )
            
            customer_type = st.selectbox(
                "Customer Type",
                options=["Transient", "Contract", "Transient-Party", "Group"],
                index=0,
                help="Type of booking"
            )
        
        with col10:
            is_repeated_guest = st.selectbox(
                "Repeated Guest",
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                index=0,
                help="Is the customer a repeated guest?"
            )
            
            previous_cancellations = st.number_input(
                "Previous Cancellations",
                min_value=0,
                max_value=26,
                value=0,
                step=1,
                help="Number of previous bookings cancelled by the customer"
            )
            
            previous_bookings_not_canceled = st.number_input(
                "Previous Bookings Not Canceled",
                min_value=0,
                max_value=72,
                value=0,
                step=1,
                help="Number of previous bookings not cancelled by the customer"
            )
        
        # Section 5: Room & Deposit Information
        st.markdown("### 🔑 Room & Payment Details")
        col11, col12 = st.columns(2)
        
        with col11:
            reserved_room_type = st.selectbox(
                "Reserved Room Type",
                options=["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"],
                index=0,
                help="Code of room type reserved"
            )
            
            assigned_room_type = st.selectbox(
                "Assigned Room Type",
                options=["A", "B", "C", "D", "E", "F", "G", "H", "L", "P"],
                index=0,
                help="Code of room type assigned (can differ from reserved)"
            )
        
        with col12:
            deposit_type = st.selectbox(
                "Deposit Type",
                options=["No Deposit", "Refundable", "Non Refund"],
                index=0,
                help="Type of deposit made"
            )
            
            booking_changes = st.number_input(
                "Booking Changes",
                min_value=0,
                max_value=21,
                value=0,
                step=1,
                help="Number of changes made to the booking"
            )
        
        # Section 6: Additional Information
        st.markdown("### ➕ Additional Details")
        col13, col14, col15 = st.columns(3)
        
        with col13:
            days_in_waiting_list = st.number_input(
                "Days in Waiting List",
                min_value=0,
                max_value=391,
                value=0,
                step=1,
                help="Number of days the booking was in the waiting list"
            )
        
        with col14:
            required_car_parking_spaces = st.number_input(
                "Parking Spaces Required",
                min_value=0,
                max_value=8,
                value=0,
                step=1,
                help="Number of car parking spaces required"
            )
        
        with col15:
            total_of_special_requests = st.number_input(
                "Special Requests",
                min_value=0,
                max_value=5,
                value=0,
                step=1,
                help="Number of special requests made"
            )
        
        # Country selection (separate row due to many options)
        country = st.selectbox(
            "Country",
            options=["PRT", "GBR", "USA", "ESP", "IRL", "FRA", "DEU", "ITA", 
                    "BEL", "NLD", "CHE", "AUT", "POL", "SWE", "NOR", "DNK",
                    "FIN", "CZE", "ROU", "RUS", "BRA", "CHN", "JPN", "Other"],
            index=0,
            help="Country of origin (ISO 3166-3 code)"
        )
        
        # Arrival day of month
        arrival_date_day_of_month = st.slider(
            "Arrival Day of Month",
            min_value=1,
            max_value=31,
            value=15,
            help="Day of the month for arrival"
        )
        
        # Submit button
        st.markdown("---")
        submit_button = st.form_submit_button(
            label="🔮 Predict Cancellation",
            use_container_width=True,
            type="primary"
        )
    
    # Return booking data if form is submitted
    if submit_button:
        booking_data = {
            "hotel": hotel,
            "lead_time": lead_time,
            "arrival_date_year": arrival_date_year,
            "arrival_date_month": arrival_date_month,
            "arrival_date_week_number": arrival_date_week_number,
            "arrival_date_day_of_month": arrival_date_day_of_month,
            "stays_in_weekend_nights": stays_in_weekend_nights,
            "stays_in_week_nights": stays_in_week_nights,
            "adults": adults,
            "children": children,
            "babies": babies,
            "meal": meal,
            "country": country,
            "market_segment": market_segment,
            "distribution_channel": distribution_channel,
            "is_repeated_guest": is_repeated_guest,
            "previous_cancellations": previous_cancellations,
            "previous_bookings_not_canceled": previous_bookings_not_canceled,
            "reserved_room_type": reserved_room_type,
            "assigned_room_type": assigned_room_type,
            "booking_changes": booking_changes,
            "deposit_type": deposit_type,
            "days_in_waiting_list": days_in_waiting_list,
            "customer_type": customer_type,
            "adr": adr,
            "required_car_parking_spaces": required_car_parking_spaces,
            "total_of_special_requests": total_of_special_requests
        }
        
        return booking_data
    
    return None


def render_quick_input_form() -> Dict[str, Any]:
    """
    Render a simplified quick input form with only the most important features.
    
    This is a streamlined version focusing on the top predictive features
    for faster user input when detailed information is not available.
    
    Returns:
        dict: Dictionary containing essential user inputs with feature names as keys.
              Returns None if form is not submitted.
    """
    
    st.subheader("⚡ Quick Prediction")
    st.markdown("Enter key booking details for a fast prediction.")
    
    with st.form(key="quick_booking_form"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            hotel = st.selectbox("Hotel Type", ["Resort Hotel", "City Hotel"], index=1)
            lead_time = st.number_input("Lead Time (days)", 0, 737, 100, 1)
            adr = st.number_input("Average Daily Rate", 0.0, 5400.0, 100.0, 5.0)
            deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Refundable", "Non Refund"], index=0)
        
        with col2:
            total_nights = st.number_input("Total Nights", 1, 50, 3, 1)
            adults = st.number_input("Adults", 1, 55, 2, 1)
            previous_cancellations = st.number_input("Previous Cancellations", 0, 26, 0, 1)
            customer_type = st.selectbox("Customer Type", ["Transient", "Contract", "Transient-Party", "Group"], index=0)
        
        submit_button = st.form_submit_button("🔮 Quick Predict", use_container_width=True, type="primary")
    
    if submit_button:
        # Create full booking data with defaults for missing fields
        booking_data = {
            "hotel": hotel,
            "lead_time": lead_time,
            "arrival_date_year": datetime.now().year,
            "arrival_date_month": "July",
            "arrival_date_week_number": 27,
            "arrival_date_day_of_month": 15,
            "stays_in_weekend_nights": min(2, total_nights),
            "stays_in_week_nights": max(0, total_nights - 2),
            "adults": adults,
            "children": 0,
            "babies": 0,
            "meal": "BB",
            "country": "PRT",
            "market_segment": "Online TA",
            "distribution_channel": "TA/TO",
            "is_repeated_guest": 0,
            "previous_cancellations": previous_cancellations,
            "previous_bookings_not_canceled": 0,
            "reserved_room_type": "A",
            "assigned_room_type": "A",
            "booking_changes": 0,
            "deposit_type": deposit_type,
            "days_in_waiting_list": 0,
            "customer_type": customer_type,
            "adr": adr,
            "required_car_parking_spaces": 0,
            "total_of_special_requests": 0
        }
        
        return booking_data
    
    return None
