# Model Information Page

## Overview
The Model Information page provides comprehensive details about the loaded machine learning model, including its performance metrics, configuration, and feature importance.

## Features Implemented

### 1. Model Overview
- Displays model type (e.g., RandomForestClassifier, XGBoostClassifier)
- Shows model version
- Displays training date

### 2. How the Model Works
An expandable section that explains in simple terms:
- What the model is and how it works
- How predictions are made
- What factors influence cancellation predictions
- Model accuracy and reliability
- Why machine learning is used for this task

### 3. Performance Metrics
Displays key performance indicators in metric cards:
- **Accuracy**: Overall prediction accuracy
- **F1-Score**: Balanced measure of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Precision**: Accuracy of positive predictions

### 4. Model Configuration
- Expandable section showing hyperparameters used during training
- Displayed as JSON for easy inspection

### 5. Training Information
- Number of training samples
- Model file path
- Class distribution in training data

### 6. Feature Importance
Displays the top 10 most important features in two formats:
- **Table**: Ranked list with importance scores
- **Chart**: Horizontal bar chart with color-coded importance values

### 7. Model Status
- Indicates if model is loaded successfully
- Shows preprocessor status

## Usage

The Model Information page is accessible from the sidebar navigation menu. Simply select "Model Info" to view all model details.

## Requirements

This page requires:
- A trained model saved with metadata (using ModelRegistry)
- Model metadata including metrics, hyperparameters, and feature importance
- Plotly for interactive visualizations
- Streamlit for the web interface

## Technical Details

The page retrieves model information through the `PredictionService.get_model_info()` method, which returns a dictionary containing all model metadata stored during training.
