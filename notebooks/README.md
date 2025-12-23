# Notebooks Directory

This directory contains Jupyter notebooks documenting the hotel cancellation prediction project.

## Notebooks

### 01_data_exploration.ipynb
Comprehensive exploratory data analysis including:
- Dataset overview and statistics
- Feature distributions and correlations
- Class imbalance analysis
- Outlier detection
- Key insights and patterns

### 03_model_training.ipynb
Model training and evaluation including:
- Loading processed data
- Class imbalance handling with SMOTE
- Training multiple models (Logistic Regression, Random Forest, XGBoost)
- Cross-validation results
- Model comparison and selection
- Model registry and versioning

### 04_model_optimization.ipynb
Hyperparameter optimization including:
- Loading best performing model
- Defining parameter grids
- RandomizedSearchCV optimization
- Performance improvement verification
- Optimized model evaluation and saving

### 05_final_report.ipynb
**Comprehensive final project report including:**
- Executive summary and key achievements
- Project objectives and methodology
- Data exploration findings
- Preprocessing and feature engineering decisions
- Model training and comparison results
- Feature importance analysis
- Hyperparameter optimization results
- Final model performance metrics
- Confusion matrix and ROC curve visualizations
- Limitations and challenges
- Future improvements
- Business impact and recommendations
- Conclusion and project summary

## Usage

To run the notebooks:

```bash
# Navigate to the notebooks directory
cd notebooks

# Start Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

## Requirements

All notebooks require the dependencies listed in `requirements.txt` at the project root.

## Execution Order

For a complete walkthrough of the project:

1. Start with `01_data_exploration.ipynb` to understand the data
2. Review `03_model_training.ipynb` for model development
3. Check `04_model_optimization.ipynb` for hyperparameter tuning
4. Read `05_final_report.ipynb` for comprehensive project summary

## Notes

- All notebooks assume the project structure is intact
- Data files should be in the `data/` directory
- Models are saved in the `models/` directory
- Visualizations are saved in `reports/figures/`
