"""
Hotel Cancellation Predictor - Streamlit Web Application

This is the main application file for the hotel cancellation prediction system.
It provides an interactive web interface for users to input booking details
and receive cancellation probability predictions.
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.prediction.prediction_service import PredictionService
from src.prediction.input_validator import InputValidator
from src.utils.exceptions import PredictionError
from src.utils.logger import get_logger

logger = get_logger(__name__)


# Page configuration
st.set_page_config(
    page_title="Hotel Cancellation Predictor",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "# Hotel Cancellation Predictor\nML-powered prediction system for hotel booking cancellations."
    }
)


@st.cache_resource
def load_prediction_service():
    """
    Load and cache the prediction service.
    
    This function is cached to avoid reloading the model on every interaction.
    The model is loaded once and reused across all user sessions.
    
    Returns:
        PredictionService: Initialized prediction service with loaded model
        
    Raises:
        Exception: If model loading fails
    """
    try:
        logger.info("Initializing prediction service...")
        
        # Try to load the best model
        model_path = "models/best_model.pkl"
        preprocessor_path = "models/preprocessor.pkl"
        
        # Check if model file exists
        if not Path(model_path).exists():
            # Try alternative model paths with version numbers
            alternative_paths = [
                "models/best_model_v1.0.0.pkl",
                "models/xgboost_v1.0.0.pkl",
                "models/random_forest_v1.0.0.pkl",
                "models/logistic_regression_v1.0.0.pkl",
                "models/xgboost_v1.pkl",
                "models/random_forest_v1.pkl",
                "models/logistic_regression_v1.pkl"
            ]
            
            for alt_path in alternative_paths:
                if Path(alt_path).exists():
                    model_path = alt_path
                    logger.info(f"Using alternative model: {model_path}")
                    break
            else:
                raise FileNotFoundError(
                    "No trained model found. Please train a model first by running the training pipeline."
                )
        
        # Check if preprocessor exists (optional)
        if not Path(preprocessor_path).exists():
            logger.warning(f"Preprocessor not found at {preprocessor_path}, will use model without preprocessing")
            preprocessor_path = None
        
        # Initialize prediction service with feature engineer
        service = PredictionService(
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            feature_engineer_dir="data/processed"
        )
        
        logger.info("Prediction service initialized successfully")
        return service
        
    except Exception as e:
        logger.error(f"Failed to initialize prediction service: {str(e)}")
        raise


def show_error_page(error_message: str):
    """
    Display a user-friendly error page when model loading fails.
    
    Args:
        error_message: The error message to display
    """
    st.error("⚠️ Application Initialization Error")
    
    st.markdown("""
    ### Unable to Load Prediction Model
    
    The application could not load the required machine learning model.
    This usually happens when:
    
    1. **No model has been trained yet** - You need to train a model first
    2. **Model files are missing** - The model files may have been deleted or moved
    3. **Incorrect file paths** - The model path in the configuration is incorrect
    """)
    
    with st.expander("📋 Error Details"):
        st.code(error_message)
    
    st.markdown("""
    ### 🔧 How to Fix This
    
    **Option 1: Train a Model**
    ```bash
    # Run the training pipeline to create a model
    python run_pipeline.py
    ```
    
    **Option 2: Run Training Notebooks**
    ```bash
    # Open and run the training notebooks in order:
    # 1. notebooks/01_data_exploration.ipynb
    # 2. notebooks/03_model_training.ipynb
    # 3. notebooks/04_model_optimization.ipynb
    ```
    
    **Option 3: Check Model Files**
    - Ensure model files exist in the `models/` directory
    - Expected files: `best_model.pkl` or `xgboost_v1.pkl`
    - Optional: `preprocessor.pkl` for data preprocessing
    
    After fixing the issue, refresh this page to try again.
    """)
    
    if st.button("🔄 Retry Loading Model"):
        st.cache_resource.clear()
        st.rerun()


def render_sidebar():
    """
    Render the sidebar with navigation menu and information.
    
    Returns:
        str: Selected page name
    """
    with st.sidebar:
        st.title("🏨 Hotel Cancellation Predictor")
        st.markdown("---")
        
        # Navigation menu
        st.subheader("📍 Navigation")
        page = st.radio(
            "Select a page:",
            options=["Prediction", "Model Info", "Batch Prediction"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Information section
        st.subheader("ℹ️ About")
        st.markdown("""
        This application uses machine learning to predict the probability 
        of hotel booking cancellations.
        
        **Features:**
        - Real-time predictions
        - Risk level assessment
        - Feature importance analysis
        - Batch predictions from CSV
        """)
        
        st.markdown("---")
        
        # Model status
        st.subheader("📊 Model Status")
        try:
            service = st.session_state.get('prediction_service')
            if service:
                model_info = service.get_model_info()
                st.success("✅ Model Loaded")
                st.caption(f"Type: {model_info.get('model_type', 'Unknown')}")
                st.caption(f"Version: {model_info.get('version', 'N/A')}")
            else:
                st.warning("⚠️ Model Not Loaded")
        except Exception as e:
            st.error("❌ Model Error")
            st.caption(str(e)[:50] + "...")
        
        return page


def render_prediction_page():
    """
    Render the main prediction page with input form and results.
    
    This page integrates:
    - Input form for booking details
    - Input validation
    - Prediction service
    - Prediction result display
    - Feature importance visualization
    """
    st.title("🔮 Cancellation Prediction")
    st.markdown("""
    Enter booking details below to predict the probability of cancellation.
    The model will analyze the booking characteristics and provide a risk assessment.
    """)
    
    st.info("💡 **Tip:** Fill in the form with booking details and click 'Predict Cancellation' to get results.")
    
    # Import components
    from app.components.input_form import render_input_form
    from app.components.prediction_display import display_prediction_result
    from app.components.visualizations import plot_probability_gauge, plot_feature_importance
    
    # Render input form
    booking_data = render_input_form()
    
    # Process prediction if form is submitted
    if booking_data:
        with st.spinner("🔄 Analyzing booking data..."):
            try:
                # Get prediction service
                service = st.session_state.get('prediction_service')
                
                if not service:
                    st.error("❌ Prediction service not available. Please refresh the page.")
                    return
                
                # Validate inputs using InputValidator
                validator = InputValidator(strict_mode=False)
                is_valid, error_message = validator.validate_booking_data(booking_data)
                
                if not is_valid:
                    st.error(f"❌ Validation Error: {error_message}")
                    logger.warning(f"Input validation failed: {error_message}")
                    
                    # Show validation details in expander
                    with st.expander("🔍 Validation Details"):
                        st.markdown(f"**Error:** {error_message}")
                        st.markdown("**Please check your input and try again.**")
                    return
                
                # Make prediction
                logger.info("Making prediction for validated booking data")
                result = service.predict(booking_data)
                
                # Display success message
                st.success("✅ Prediction Complete!")
                
                # Create two columns for layout
                col_left, col_right = st.columns([1, 1])
                
                with col_left:
                    # Display prediction results using the prediction display component
                    display_prediction_result(result)
                
                with col_right:
                    # Show probability gauge visualization
                    st.markdown("### 📊 Probability Gauge")
                    probability = result.get('probability', 0.0)
                    gauge_fig = plot_probability_gauge(probability)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Show feature importance visualization
                st.markdown("---")
                st.markdown("### 🎯 Feature Importance")
                st.markdown("""
                The chart below shows the top features that contribute most to cancellation predictions.
                These are the factors the model considers most important when making decisions.
                """)
                
                # Get model info to extract feature importance
                model_info = service.get_model_info()
                feature_importance = model_info.get('feature_importance', {})
                
                if feature_importance:
                    # Plot feature importance
                    importance_fig = plot_feature_importance(
                        feature_importance,
                        top_n=10,
                        title="Top 10 Features Influencing This Prediction"
                    )
                    st.plotly_chart(importance_fig, use_container_width=True)
                    
                    # Add explanation
                    with st.expander("ℹ️ Understanding Feature Importance"):
                        st.markdown("""
                        **What is Feature Importance?**
                        
                        Feature importance shows which booking characteristics have the most influence 
                        on the model's predictions. Higher importance means the feature has a stronger 
                        impact on whether a booking is predicted to be cancelled or not.
                        
                        **Common Important Features:**
                        - **Lead Time**: How far in advance the booking was made
                        - **ADR (Average Daily Rate)**: The price per night
                        - **Deposit Type**: Whether a deposit was required
                        - **Previous Cancellations**: Guest's history of cancellations
                        - **Market Segment**: How the booking was made (online, direct, etc.)
                        
                        **How to Use This Information:**
                        - Features at the top have the most influence on predictions
                        - Understanding these factors can help you identify high-risk bookings
                        - You can focus on these key features when reviewing bookings manually
                        """)
                else:
                    st.info("Feature importance information is not available for this model.")
                
                # Add option to make another prediction
                st.markdown("---")
                if st.button("🔄 Make Another Prediction", use_container_width=True):
                    st.rerun()
                
            except PredictionError as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                logger.error(f"Prediction failed: {str(e)}")
                
                # Show error details
                with st.expander("🔍 Error Details"):
                    st.code(str(e))
                    st.markdown("""
                    **Possible causes:**
                    - Model file is corrupted or incompatible
                    - Preprocessing failed due to unexpected data format
                    - Model requires features that are not provided
                    
                    **What to do:**
                    - Check that all required fields are filled correctly
                    - Try refreshing the page
                    - If the problem persists, the model may need to be retrained
                    """)
            
            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}")
                logger.error(f"Unexpected error during prediction: {str(e)}", exc_info=True)
                
                # Show error details
                with st.expander("🔍 Error Details"):
                    st.code(str(e))
                    st.markdown("""
                    **An unexpected error occurred.**
                    
                    Please try the following:
                    1. Refresh the page and try again
                    2. Check that all input values are reasonable
                    3. If the problem persists, contact support
                    """)


def render_model_info_page():
    """
    Render the model information page with performance metrics and details.
    """
    st.title("📊 Model Information")
    st.markdown("View details about the loaded prediction model and its performance.")
    
    try:
        service = st.session_state.get('prediction_service')
        
        if not service:
            st.warning("Prediction service not available.")
            return
        
        model_info = service.get_model_info()
        
        # Model Overview
        st.subheader("🤖 Model Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model Type", model_info.get('model_type', 'Unknown'))
        
        with col2:
            st.metric("Version", model_info.get('version', 'N/A'))
        
        with col3:
            st.metric("Training Date", model_info.get('training_date', 'N/A')[:10] if model_info.get('training_date') else 'N/A')
        
        # How the Model Works
        st.subheader("🧠 How the Model Works")
        
        with st.expander("📖 Understanding the Prediction Model", expanded=False):
            st.markdown("""
            ### What is this model?
            
            This is a **machine learning model** trained to predict whether a hotel booking will be cancelled or not. 
            Think of it as a smart assistant that has learned patterns from thousands of past bookings.
            
            ### How does it make predictions?
            
            1. **Learning from History**: The model was trained on historical booking data, learning which characteristics 
               are commonly associated with cancellations. For example, it learned that bookings with very long lead times 
               or certain deposit types tend to have higher cancellation rates.
            
            2. **Analyzing New Bookings**: When you input a new booking, the model examines all its features 
               (lead time, price, guest count, etc.) and compares them to the patterns it learned during training.
            
            3. **Calculating Probability**: Instead of just saying "yes" or "no", the model calculates a probability 
               score between 0% and 100%. A score of 75% means the model is 75% confident this booking will be cancelled.
            
            4. **Risk Assessment**: Based on the probability, bookings are categorized into risk levels:
               - **Low Risk** (< 30%): Unlikely to cancel
               - **Medium Risk** (30-70%): Moderate chance of cancellation
               - **High Risk** (> 70%): Likely to cancel
            
            ### What makes a booking likely to cancel?
            
            The model considers many factors, but the most important ones typically include:
            - **Lead Time**: How far in advance the booking was made
            - **Average Daily Rate (ADR)**: The price per night
            - **Deposit Type**: Whether a deposit was required
            - **Previous Cancellations**: Guest's cancellation history
            - **Market Segment**: How the booking was made (online, travel agent, etc.)
            - **Special Requests**: Number of special requests made
            
            ### How accurate is it?
            
            The model's accuracy is shown in the Performance Metrics section below. An F1-score above 0.75 indicates 
            good predictive performance. However, no model is perfect - it provides probabilities to help you make 
            informed decisions, not absolute certainties.
            
            ### Why use machine learning?
            
            Traditional rule-based systems can't capture the complex interactions between different booking features. 
            Machine learning excels at finding subtle patterns in large datasets that humans might miss, making it 
            ideal for this type of prediction task.
            """)
        
        # Performance Metrics
        st.subheader("📈 Performance Metrics")
        
        metrics = model_info.get('metrics', {})
        
        if metrics:
            col4, col5, col6, col7 = st.columns(4)
            
            with col4:
                st.metric("Accuracy", f"{metrics.get('accuracy', 0)*100:.2f}%")
            
            with col5:
                st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")
            
            with col6:
                st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
            
            with col7:
                st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        else:
            st.info("No performance metrics available for this model.")
        
        # Model Configuration
        st.subheader("⚙️ Model Configuration")
        
        hyperparameters = model_info.get('hyperparameters', {})
        
        if hyperparameters:
            with st.expander("View Hyperparameters"):
                st.json(hyperparameters)
        else:
            st.info("No hyperparameter information available.")
        
        # Training Information
        st.subheader("📚 Training Information")
        
        col8, col9 = st.columns(2)
        
        with col8:
            st.write(f"**Training Samples:** {model_info.get('training_samples', 'N/A')}")
            st.write(f"**Model Path:** `{model_info.get('model_path', 'N/A')}`")
        
        with col9:
            class_dist = model_info.get('class_distribution', {})
            if class_dist:
                st.write("**Class Distribution:**")
                for class_label, count in class_dist.items():
                    st.write(f"  - Class {class_label}: {count}")
            else:
                st.write("**Class Distribution:** N/A")
        
        # Feature Importance
        st.subheader("🎯 Feature Importance")
        
        feature_importance = model_info.get('feature_importance', {})
        
        if feature_importance:
            # Sort by importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]  # Top 10 features
            
            # Create two columns for table and chart
            col_table, col_chart = st.columns([1, 1])
            
            with col_table:
                st.markdown("**Top 10 Most Important Features:**")
                
                # Display as a formatted table
                import pandas as pd
                features_df = pd.DataFrame(
                    sorted_features,
                    columns=['Feature', 'Importance']
                )
                features_df.index = range(1, len(features_df) + 1)
                st.dataframe(features_df, use_container_width=True)
            
            with col_chart:
                st.markdown("**Feature Importance Chart:**")
                
                # Create bar chart
                import plotly.graph_objects as go
                
                feature_names = [f[0] for f in sorted_features]
                importance_values = [f[1] for f in sorted_features]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=importance_values,
                        y=feature_names,
                        orientation='h',
                        marker=dict(
                            color=importance_values,
                            colorscale='Blues',
                            showscale=False
                        )
                    )
                ])
                
                fig.update_layout(
                    xaxis_title="Importance Score",
                    yaxis_title="Feature",
                    height=400,
                    margin=dict(l=0, r=0, t=0, b=0),
                    yaxis=dict(autorange="reversed")
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance information not available for this model.")
        
        # Model Status
        st.subheader("✅ Model Status")
        
        col10, col11 = st.columns(2)
        
        with col10:
            if model_info.get('model_loaded'):
                st.success("✅ Model Loaded Successfully")
            else:
                st.error("❌ Model Not Loaded")
        
        with col11:
            if model_info.get('preprocessor_loaded'):
                st.success("✅ Preprocessor Loaded")
            else:
                st.warning("⚠️ No Preprocessor")
        
    except Exception as e:
        st.error(f"Error loading model information: {str(e)}")
        logger.error(f"Error in model info page: {str(e)}")


def render_batch_prediction_page():
    """
    Render the batch prediction page for processing multiple bookings from CSV.
    """
    st.title("📦 Batch Prediction")
    st.markdown("""
    Upload a CSV file with multiple booking records to get predictions for all of them at once.
    This is useful for processing large numbers of bookings efficiently.
    """)
    
    # Required columns for validation
    REQUIRED_COLUMNS = [
        'hotel', 'lead_time', 'arrival_date_year', 'arrival_date_month', 
        'arrival_date_week_number', 'arrival_date_day_of_month', 
        'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 
        'children', 'babies', 'meal', 'country', 'market_segment',
        'distribution_channel', 'is_repeated_guest', 'previous_cancellations',
        'previous_bookings_not_canceled', 'reserved_room_type', 
        'assigned_room_type', 'booking_changes', 'deposit_type', 
        'days_in_waiting_list', 'customer_type', 'adr', 
        'required_car_parking_spaces', 'total_of_special_requests'
    ]
    
    # Instructions
    with st.expander("📋 CSV Format Instructions"):
        st.markdown("""
        Your CSV file should contain the following columns:
        
        **Required Columns:**
        - `hotel`, `lead_time`, `arrival_date_year`, `arrival_date_month`, `arrival_date_week_number`
        - `arrival_date_day_of_month`, `stays_in_weekend_nights`, `stays_in_week_nights`
        - `adults`, `children`, `babies`, `meal`, `country`, `market_segment`
        - `distribution_channel`, `is_repeated_guest`, `previous_cancellations`
        - `previous_bookings_not_canceled`, `reserved_room_type`, `assigned_room_type`
        - `booking_changes`, `deposit_type`, `days_in_waiting_list`, `customer_type`
        - `adr`, `required_car_parking_spaces`, `total_of_special_requests`
        
        **Note:** All columns from the training dataset should be present. You can use the sample data 
        from `data/raw/hotel_bookings.csv` as a reference for the correct format.
        
        **Example:**
        ```csv
        hotel,lead_time,arrival_date_year,arrival_date_month,...
        Resort Hotel,342,2015,July,...
        City Hotel,737,2015,July,...
        ```
        """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload a CSV file containing booking data"
    )
    
    if uploaded_file is not None:
        try:
            import pandas as pd
            
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} bookings.")
            
            # Validate required columns
            missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                st.info("Please ensure your CSV file contains all required columns. See the format instructions above.")
                return
            
            # Show column validation success
            st.success(f"✅ All {len(REQUIRED_COLUMNS)} required columns found!")
            
            # Show preview
            with st.expander("👀 Preview Data (First 10 rows)"):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Show data info
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Total Rows", len(df))
            with col_info2:
                st.metric("Total Columns", len(df.columns))
            with col_info3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Predict button
            if st.button("🔮 Predict All Bookings", type="primary"):
                with st.spinner(f"Processing {len(df)} bookings..."):
                    try:
                        service = st.session_state.get('prediction_service')
                        
                        if not service:
                            st.error("Prediction service not available.")
                            return
                        
                        # Convert dataframe to list of dicts
                        bookings = df.to_dict('records')
                        
                        # Make batch predictions
                        results = service.predict_batch(bookings)
                        
                        # Create results dataframe
                        results_df = pd.DataFrame(results)
                        
                        # Check for errors in results
                        error_count = sum(1 for r in results if r.get('error') is not None)
                        success_count = len(results) - error_count
                        
                        # Display summary
                        if error_count > 0:
                            st.warning(f"⚠️ Batch prediction completed with {error_count} errors. {success_count} predictions successful.")
                        else:
                            st.success("✅ Batch prediction complete! All bookings processed successfully.")
                        
                        # Summary statistics
                        st.subheader("📈 Summary Statistics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Bookings", len(results))
                        
                        with col2:
                            predicted_cancellations = sum(
                                1 for r in results if r.get('prediction') == 1
                            )
                            cancellation_rate = (predicted_cancellations / success_count * 100) if success_count > 0 else 0
                            st.metric(
                                "Predicted Cancellations", 
                                predicted_cancellations,
                                delta=f"{cancellation_rate:.1f}%"
                            )
                        
                        with col3:
                            # Calculate average probability only for successful predictions
                            valid_probabilities = [
                                r.get('probability', 0) for r in results 
                                if r.get('probability') is not None and r.get('error') is None
                            ]
                            avg_probability = sum(valid_probabilities) / len(valid_probabilities) if valid_probabilities else 0
                            st.metric("Avg Cancellation Probability", f"{avg_probability*100:.1f}%")
                        
                        with col4:
                            high_risk_count = sum(
                                1 for r in results if r.get('risk_level') == 'high'
                            )
                            st.metric("High Risk Bookings", high_risk_count)
                        
                        # Risk distribution
                        st.subheader("🎯 Risk Distribution")
                        
                        risk_counts = {
                            'low': sum(1 for r in results if r.get('risk_level') == 'low'),
                            'medium': sum(1 for r in results if r.get('risk_level') == 'medium'),
                            'high': sum(1 for r in results if r.get('risk_level') == 'high')
                        }
                        
                        col_risk1, col_risk2, col_risk3 = st.columns(3)
                        
                        with col_risk1:
                            st.metric("🟢 Low Risk", risk_counts['low'])
                        with col_risk2:
                            st.metric("🟡 Medium Risk", risk_counts['medium'])
                        with col_risk3:
                            st.metric("🔴 High Risk", risk_counts['high'])
                        
                        # Display results table
                        st.subheader("📊 Detailed Prediction Results")
                        
                        # Prepare display dataframe
                        display_columns = ['booking_index', 'prediction', 'probability', 'risk_level']
                        
                        # Add error column if there are errors
                        if error_count > 0:
                            display_columns.append('error')
                        
                        # Filter to only display columns that exist
                        display_columns = [col for col in display_columns if col in results_df.columns]
                        display_df = results_df[display_columns]
                        
                        # Format probability as percentage
                        if 'probability' in display_df.columns:
                            display_df['probability'] = display_df['probability'].apply(
                                lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A"
                            )
                        
                        # Format prediction as text
                        if 'prediction' in display_df.columns:
                            display_df['prediction'] = display_df['prediction'].apply(
                                lambda x: "Will Cancel" if x == 1 else ("Will Not Cancel" if x == 0 else "Error")
                            )
                        
                        # Display with pagination
                        st.dataframe(
                            display_df, 
                            use_container_width=True,
                            height=400
                        )
                        
                        # Download section
                        st.subheader("💾 Export Results")
                        
                        col_download1, col_download2 = st.columns(2)
                        
                        with col_download1:
                            # Download full results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Full Results (CSV)",
                                data=csv,
                                file_name=f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        with col_download2:
                            # Download only high-risk bookings
                            high_risk_df = results_df[results_df['risk_level'] == 'high']
                            if not high_risk_df.empty:
                                high_risk_csv = high_risk_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download High Risk Only (CSV)",
                                    data=high_risk_csv,
                                    file_name=f"high_risk_bookings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            else:
                                st.info("No high-risk bookings to download")
                        
                        # Show errors if any
                        if error_count > 0:
                            with st.expander(f"⚠️ View Errors ({error_count} bookings failed)"):
                                error_df = results_df[results_df['error'].notna()][['booking_index', 'error']]
                                st.dataframe(error_df, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error during batch prediction: {str(e)}")
                        logger.error(f"Batch prediction error: {str(e)}")
                        
                        with st.expander("🔍 Error Details"):
                            st.code(str(e))
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            logger.error(f"CSV reading error: {str(e)}")
            
            with st.expander("🔍 Error Details"):
                st.code(str(e))


def main():
    """
    Main application entry point.
    """
    try:
        # Initialize prediction service and store in session state
        if 'prediction_service' not in st.session_state:
            st.session_state.prediction_service = load_prediction_service()
        
        # Render sidebar and get selected page
        selected_page = render_sidebar()
        
        # Render selected page
        if selected_page == "Prediction":
            render_prediction_page()
        elif selected_page == "Model Info":
            render_model_info_page()
        elif selected_page == "Batch Prediction":
            render_batch_prediction_page()
    
    except Exception as e:
        # Show error page if initialization fails
        show_error_page(str(e))


if __name__ == "__main__":
    main()
