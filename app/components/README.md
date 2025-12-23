# Web Application Components

This directory contains reusable Streamlit components for the Hotel Cancellation Prediction web interface.

## Components

### 1. input_form.py

Provides interactive input forms for collecting booking details from users.

#### Functions

##### `render_input_form() -> Dict[str, Any]`

Renders a comprehensive input form with all booking features required for prediction.

**Features:**
- Organized into logical sections (Basic Info, Stay Details, Guest Info, etc.)
- Uses appropriate Streamlit widgets for each field type
- Includes helpful tooltips and descriptions
- Sets sensible default values based on data exploration
- Validates input ranges based on dataset statistics

**Returns:**
- `dict`: Dictionary with all booking features when form is submitted
- `None`: When form is not yet submitted

**Example Usage:**
```python
import streamlit as st
from app.components.input_form import render_input_form

st.title("Hotel Cancellation Predictor")
booking_data = render_input_form()

if booking_data:
    # Process the booking data
    prediction = predict(booking_data)
    st.write(f"Cancellation Probability: {prediction['probability']:.2%}")
```

##### `render_quick_input_form() -> Dict[str, Any]`

Renders a simplified form with only the most important predictive features.

**Features:**
- Streamlined interface for faster input
- Focuses on top predictive features
- Automatically fills in defaults for non-essential fields
- Ideal for quick predictions or mobile interfaces

**Returns:**
- `dict`: Dictionary with all booking features (essential + defaults) when submitted
- `None`: When form is not yet submitted

**Example Usage:**
```python
from app.components.input_form import render_quick_input_form

booking_data = render_quick_input_form()

if booking_data:
    # booking_data contains all required fields with defaults
    prediction = predict(booking_data)
```

## Input Fields

### Full Form Fields

The full form includes all 24+ booking features:

**Basic Information:**
- Hotel Type (Resort/City)
- Lead Time (0-737 days)
- Average Daily Rate (0-5400)
- Arrival Date (Year, Month, Week, Day)

**Stay Details:**
- Weekend Nights (0-19)
- Week Nights (0-50)
- Meal Plan (BB/HB/FB/SC/Undefined)

**Guest Information:**
- Adults (0-55)
- Children (0-10)
- Babies (0-10)

**Booking Details:**
- Market Segment
- Distribution Channel
- Customer Type
- Repeated Guest (Yes/No)
- Previous Cancellations (0-26)
- Previous Bookings Not Canceled (0-72)

**Room & Payment:**
- Reserved Room Type (A-P)
- Assigned Room Type (A-P)
- Deposit Type (No Deposit/Refundable/Non Refund)
- Booking Changes (0-21)

**Additional Details:**
- Days in Waiting List (0-391)
- Parking Spaces Required (0-8)
- Special Requests (0-5)
- Country (ISO codes)

### Quick Form Fields

The quick form includes only essential features:
- Hotel Type
- Lead Time
- Average Daily Rate
- Deposit Type
- Total Nights
- Number of Adults
- Previous Cancellations
- Customer Type

All other fields are automatically filled with sensible defaults.

## Default Values

Default values are based on insights from exploratory data analysis:

- **Lead Time**: 100 days (median value)
- **ADR**: 100.0 (approximate median)
- **Adults**: 2 (most common)
- **Nights**: 3 total (1 weekend + 2 week nights)
- **Meal**: BB (Bed & Breakfast, most common)
- **Market Segment**: Online TA (most common)
- **Deposit Type**: No Deposit (most common)
- **Customer Type**: Transient (most common)

## Value Ranges

Ranges are set based on actual data distribution:

- **Lead Time**: 0-737 days (max observed in dataset)
- **ADR**: 0-5400 (max observed, though outliers exist)
- **Weekend Nights**: 0-19 (max observed)
- **Week Nights**: 0-50 (max observed)
- **Adults**: 0-55 (max observed)
- **Previous Cancellations**: 0-26 (max observed)
- **Days in Waiting List**: 0-391 (max observed)

## Integration

To integrate these components into the main Streamlit app:

```python
# In app/streamlit_app.py

from app.components.input_form import render_input_form
from src.prediction.prediction_service import PredictionService

# Initialize prediction service
service = PredictionService(
    model_path='models/best_model.pkl',
    preprocessor_path='models/preprocessor.pkl'
)

# Render form
booking_data = render_input_form()

# Make prediction when form is submitted
if booking_data:
    try:
        result = service.predict(booking_data)
        # Display result using prediction_display component
        display_prediction_result(result)
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
```

## Testing

To test the input form component:

```bash
# Run the example script
streamlit run examples/input_form_example.py
```

This will launch a demo application showing both form types and displaying submitted data.

## Future Enhancements

Potential improvements for the input form:

1. **Dynamic Field Validation**: Real-time validation with error messages
2. **Auto-complete**: For country selection and other fields
3. **Date Picker**: Replace separate year/month/day fields with date picker
4. **Preset Templates**: Quick-fill buttons for common booking types
5. **Field Dependencies**: Auto-adjust related fields (e.g., room type match)
6. **Mobile Optimization**: Responsive layout for mobile devices
7. **Multi-language Support**: Internationalization of labels and help text
8. **Accessibility**: Enhanced ARIA labels and keyboard navigation


### 2. predicti