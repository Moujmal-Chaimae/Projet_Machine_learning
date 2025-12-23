"""
Unit tests for visualization components.

Tests the visualization functions to ensure they generate valid Plotly figures
with correct data and formatting.
"""

import pytest
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
import plotly.graph_objects as go


class TestProbabilityGauge:
    """Tests for plot_probability_gauge function."""
    
    def test_creates_valid_figure(self):
        """Test that the function returns a valid Plotly figure."""
        fig = plot_probability_gauge(0.75)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_low_risk_probability(self):
        """Test gauge with low risk probability."""
        fig = plot_probability_gauge(0.2)
        assert isinstance(fig, go.Figure)
    
    def test_medium_risk_probability(self):
        """Test gauge with medium risk probability."""
        fig = plot_probability_gauge(0.5)
        assert isinstance(fig, go.Figure)
    
    def test_high_risk_probability(self):
        """Test gauge with high risk probability."""
        fig = plot_probability_gauge(0.85)
        assert isinstance(fig, go.Figure)
    
    def test_edge_case_zero(self):
        """Test gauge with zero probability."""
        fig = plot_probability_gauge(0.0)
        assert isinstance(fig, go.Figure)
    
    def test_edge_case_one(self):
        """Test gauge with 100% probability."""
        fig = plot_probability_gauge(1.0)
        assert isinstance(fig, go.Figure)
    
    def test_custom_title(self):
        """Test gauge with custom title."""
        custom_title = "Custom Probability Gauge"
        fig = plot_probability_gauge(0.5, title=custom_title)
        assert isinstance(fig, go.Figure)


class TestFeatureImportance:
    """Tests for plot_feature_importance function."""
    
    def test_creates_valid_figure(self):
        """Test that the function returns a valid Plotly figure."""
        importance = {
            'feature1': 0.3,
            'feature2': 0.2,
            'feature3': 0.15
        }
        fig = plot_feature_importance(importance)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_top_n_features(self):
        """Test that only top N features are displayed."""
        importance = {f'feature{i}': 0.1 - i*0.01 for i in range(20)}
        fig = plot_feature_importance(importance, top_n=5)
        assert isinstance(fig, go.Figure)
    
    def test_empty_importance(self):
        """Test with empty feature importance dictionary."""
        importance = {}
        fig = plot_feature_importance(importance)
        assert isinstance(fig, go.Figure)
    
    def test_single_feature(self):
        """Test with single feature."""
        importance = {'feature1': 0.5}
        fig = plot_feature_importance(importance)
        assert isinstance(fig, go.Figure)
    
    def test_custom_title(self):
        """Test with custom title."""
        importance = {'feature1': 0.3, 'feature2': 0.2}
        custom_title = "Custom Feature Importance"
        fig = plot_feature_importance(importance, title=custom_title)
        assert isinstance(fig, go.Figure)


class TestRiskDistribution:
    """Tests for plot_risk_distribution function."""
    
    def test_creates_valid_figure(self):
        """Test that the function returns a valid Plotly figure."""
        predictions = [
            {'risk_level': 'low', 'probability': 0.2},
            {'risk_level': 'medium', 'probability': 0.5},
            {'risk_level': 'high', 'probability': 0.8}
        ]
        fig = plot_risk_distribution(predictions)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_all_same_risk_level(self):
        """Test with all predictions having the same risk level."""
        predictions = [
            {'risk_level': 'high', 'probability': 0.8},
            {'risk_level': 'high', 'probability': 0.85},
            {'risk_level': 'high', 'probability': 0.9}
        ]
        fig = plot_risk_distribution(predictions)
        assert isinstance(fig, go.Figure)
    
    def test_empty_predictions(self):
        """Test with empty predictions list."""
        predictions = []
        fig = plot_risk_distribution(predictions)
        assert isinstance(fig, go.Figure)
    
    def test_single_prediction(self):
        """Test with single prediction."""
        predictions = [{'risk_level': 'medium', 'probability': 0.5}]
        fig = plot_risk_distribution(predictions)
        assert isinstance(fig, go.Figure)
    
    def test_custom_title(self):
        """Test with custom title."""
        predictions = [
            {'risk_level': 'low', 'probability': 0.2},
            {'risk_level': 'high', 'probability': 0.8}
        ]
        custom_title = "Custom Risk Distribution"
        fig = plot_risk_distribution(predictions, title=custom_title)
        assert isinstance(fig, go.Figure)


class TestProbabilityDistribution:
    """Tests for plot_probability_distribution function."""
    
    def test_creates_valid_figure(self):
        """Test that the function returns a valid Plotly figure."""
        predictions = [
            {'probability': 0.2},
            {'probability': 0.5},
            {'probability': 0.8}
        ]
        fig = plot_probability_distribution(predictions)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_custom_bins(self):
        """Test with custom number of bins."""
        predictions = [{'probability': i/100} for i in range(100)]
        fig = plot_probability_distribution(predictions, bins=10)
        assert isinstance(fig, go.Figure)
    
    def test_empty_predictions(self):
        """Test with empty predictions list."""
        predictions = []
        fig = plot_probability_distribution(predictions)
        assert isinstance(fig, go.Figure)


class TestFeatureContributions:
    """Tests for plot_feature_contributions function."""
    
    def test_creates_valid_figure(self):
        """Test that the function returns a valid Plotly figure."""
        booking_data = {
            'lead_time': 342,
            'adr': 95.5,
            'deposit_type': 'No Deposit'
        }
        feature_importance = {
            'lead_time': 0.3,
            'adr': 0.2,
            'deposit_type': 0.15
        }
        fig = plot_feature_contributions(booking_data, feature_importance)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_missing_booking_features(self):
        """Test when booking data is missing some features."""
        booking_data = {'lead_time': 342}
        feature_importance = {
            'lead_time': 0.3,
            'adr': 0.2,
            'missing_feature': 0.15
        }
        fig = plot_feature_contributions(booking_data, feature_importance)
        assert isinstance(fig, go.Figure)


class TestSummaryMetrics:
    """Tests for create_summary_metrics_chart function."""
    
    def test_creates_valid_figure(self):
        """Test that the function returns a valid Plotly figure."""
        predictions = [
            {'prediction': 1, 'probability': 0.8, 'risk_level': 'high'},
            {'prediction': 0, 'probability': 0.2, 'risk_level': 'low'},
            {'prediction': 1, 'probability': 0.6, 'risk_level': 'medium'}
        ]
        fig = create_summary_metrics_chart(predictions)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_empty_predictions(self):
        """Test with empty predictions list."""
        predictions = []
        fig = create_summary_metrics_chart(predictions)
        assert isinstance(fig, go.Figure)
    
    def test_all_cancellations(self):
        """Test when all predictions are cancellations."""
        predictions = [
            {'prediction': 1, 'probability': 0.8, 'risk_level': 'high'},
            {'prediction': 1, 'probability': 0.9, 'risk_level': 'high'}
        ]
        fig = create_summary_metrics_chart(predictions)
        assert isinstance(fig, go.Figure)
    
    def test_no_cancellations(self):
        """Test when no predictions are cancellations."""
        predictions = [
            {'prediction': 0, 'probability': 0.2, 'risk_level': 'low'},
            {'prediction': 0, 'probability': 0.1, 'risk_level': 'low'}
        ]
        fig = create_summary_metrics_chart(predictions)
        assert isinstance(fig, go.Figure)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
