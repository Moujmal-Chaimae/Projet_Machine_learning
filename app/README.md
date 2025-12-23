# Hotel Cancellation Predictor - Web Application

This directory contains the Streamlit web application for the Hotel Cancellation Prediction system.

## Structure

```
app/
├── streamlit_app.py          # Main application file
└── components/               # Reusable UI components
    ├── input_form.py         # Booking input form
    ├── prediction_display.py # Prediction results display (to be implemented)
    └── visualizations.py     # Charts and visualizations (to be implemented)
```

## Features

### 1. Prediction Page
- Interactive form for entering booking details
- Real-time cancellation probability prediction
- Risk level assessment (Low/Medium/High)
- Detailed prediction metrics

### 2. Model Info Page
- Model type and version information
- Performance metrics (Accuracy, F1-Score, ROC-AUC)
- Hyperparameter configuration
- Feature importance rankings
- Training information

### 3. Batch Prediction Page
- CSV file upload for multiple bookings
- Bulk prediction processing
- Summary statistics
- Downloadable results

## Running the Application

### Prerequisites

Make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

### Method 1: Using the helper script

```bash
python run_app.py
```

### Method 2: Direct Streamlit command

```bash
streamlit run app/streamlit_app.py
```

### Method 3: Custom port

```bash
streamlit run app/streamlit_app.py --server.port=8502
```

## Configuration

The application automatically:
- Loads the best available model from `models/` directory
- Caches the model to improve performance
- Handles missing models with user-friendly error messages

### Model Loading Priority

1. `models/best_model.pkl` (preferred)
2. `models/xgboost_v1.pkl`
3. `models/random_forest_v1.pkl`
4. `models/logistic_regression_v1.pkl`

## Usage

### Single Prediction

1. Navigate to the **Prediction** page
2. Fill in the booking details form
3. Click "🔮 Predict Cancellation"
4. View the prediction results and risk assessment

### Batch Prediction

1. Navigate to the **Batch Prediction** page
2. Upload a CSV file with booking data
3. Click "🔮 Predict All Bookings"
4. Download the results as CSV

### View Model Information

1. Navigate to the **Model Info** page
2. View model performance metrics
3. Check feature importance
4. Review model configuration

## Error Handling

The application includes comprehensive error handling:

- **Model not found**: Shows instructions for training a model
- **Invalid input**: Validates booking data before prediction
- **Prediction errors**: Displays user-friendly error messages
- **CSV format errors**: Provides guidance on correct format

## Troubleshooting

### Model Not Loading

If you see "Unable to Load Prediction Model":

1. **Train a model first**:
   ```bash
   python run_pipeline.py
   ```

2. **Or run the training notebooks**:
   - `notebooks/01_data_exploration.ipynb`
   - `notebooks/03_model_training.ipynb`
   - `notebooks/04_model_optimization.ipynb`

3. **Check model files exist**:
   ```bash
   ls models/
   ```

### Port Already in Use

If port 8501 is already in use:

```bash
streamlit run app/streamlit_app.py --server.port=8502
```

### Import Errors

Make sure you're running from the project root directory:

```bash
cd /path/to/hotel-cancellation-optimizer
python run_app.py
```

## Development

### Adding New Components

1. Create a new file in `app/components/`
2. Define render functions that return Streamlit components
3. Import and use in `streamlit_app.py`

Example:

```python
# app/components/my_component.py
import streamlit as st

def render_my_component():
    st.write("My custom component")
```

```python
# app/streamlit_app.py
from app.components.my_component import render_my_component

render_my_component()
```

### Modifying the Layout

The application uses Streamlit's wide layout mode. To modify:

```python
st.set_page_config(
    page_title="Your Title",
    layout="wide",  # or "centered"
    initial_sidebar_state="expanded"  # or "collapsed"
)
```

## Performance

- **Model caching**: The model is loaded once and cached using `@st.cache_resource`
- **Fast predictions**: Single predictions typically complete in < 200ms
- **Batch processing**: Efficiently handles multiple bookings

## Security Notes

- The application runs locally by default
- No sensitive data is stored
- Model files should be kept secure
- For production deployment, consider authentication and HTTPS

## Future Enhancements

- [ ] Add SHAP/LIME explanations for predictions
- [ ] Implement prediction history tracking
- [ ] Add data visualization dashboard
- [ ] Support for multiple models comparison
- [ ] Export predictions to different formats (Excel, JSON)
- [ ] Real-time model performance monitoring
