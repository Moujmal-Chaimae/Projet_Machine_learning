"""
Visualization Components for Hotel Cancellation Prediction Web Interface

This module provides interactive visualization functions using Plotly for
displaying prediction results, feature importance, and risk distributions.
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import pandas as pd


def plot_probability_gauge(probability: float, title: str = "Cancellation Probability") -> go.Figure:
    """
    Create an interactive gauge chart showing cancellation probability.
    
    The gauge uses color coding to indicate risk levels:
    - Green (0-30%): Low risk
    - Yellow (30-70%): Medium risk
    - Red (70-100%): High risk
    
    Args:
        probability: Cancellation probability between 0.0 and 1.0
        title: Title for the gauge chart (default: "Cancellation Probability")
    
    Returns:
        go.Figure: Plotly figure object with gauge chart
    
    Example:
        >>> fig = plot_probability_gauge(0.75)
        >>> fig.show()
    """
    # Convert probability to percentage
    probability_percent = probability * 100
    
    # Determine color based on risk level
    if probability < 0.3:
        gauge_color = "#00cc66"  # Green for low risk
        risk_text = "Low Risk"
    elif probability < 0.7:
        gauge_color = "#ffa500"  # Orange for medium risk
        risk_text = "Medium Risk"
    else:
        gauge_color = "#ff4b4b"  # Red for high risk
        risk_text = "High Risk"
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability_percent,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={
            'text': f"<b>{title}</b><br><span style='font-size:0.8em;color:gray'>{risk_text}</span>",
            'font': {'size': 20}
        },
        number={
            'suffix': "%",
            'font': {'size': 50}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 2,
                'tickcolor': "darkgray",
                'tickmode': 'linear',
                'tick0': 0,
                'dtick': 10
            },
            'bar': {'color': gauge_color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#e6f7ed'},      # Light green
                {'range': [30, 70], 'color': '#fff4e6'},     # Light orange
                {'range': [70, 100], 'color': '#ffe6e6'}     # Light red
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': probability_percent
            }
        }
    ))
    
    # Update layout for better appearance
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="white",
        font={'family': "Arial, sans-serif"}
    )
    
    return fig


def plot_feature_importance(
    feature_importance: Dict[str, float],
    top_n: int = 10,
    title: str = "Top 10 Most Important Features"
) -> go.Figure:
    """
    Create an interactive horizontal bar chart showing feature importance.
    
    Displays the most important features that contribute to the prediction,
    sorted by importance score in descending order.
    
    Args:
        feature_importance: Dictionary mapping feature names to importance scores
        top_n: Number of top features to display (default: 10)
        title: Title for the chart (default: "Top 10 Most Important Features")
    
    Returns:
        go.Figure: Plotly figure object with horizontal bar chart
    
    Example:
        >>> importance = {'lead_time': 0.25, 'adr': 0.20, 'deposit_type': 0.15}
        >>> fig = plot_feature_importance(importance)
        >>> fig.show()
    """
    # Sort features by importance and get top N
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    # Extract feature names and importance values
    feature_names = [f[0] for f in sorted_features]
    importance_values = [f[1] for f in sorted_features]
    
    # Reverse order for better visualization (highest at top)
    feature_names = feature_names[::-1]
    importance_values = importance_values[::-1]
    
    # Create color scale based on importance
    colors = px.colors.sequential.Blues_r[:len(importance_values)]
    
    # Create horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=importance_values,
            y=feature_names,
            orientation='h',
            marker=dict(
                color=importance_values,
                colorscale='Blues',
                showscale=False,
                line=dict(color='rgb(8,48,107)', width=1.5)
            ),
            text=[f'{val:.4f}' for val in importance_values],
            textposition='outside',
            textfont=dict(size=11),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        )
    ])
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=max(400, top_n * 40),  # Dynamic height based on number of features
        margin=dict(l=150, r=50, t=80, b=50),
        paper_bgcolor="white",
        plot_bgcolor="rgba(240, 240, 240, 0.5)",
        font={'family': "Arial, sans-serif"},
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=False
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial, sans-serif"
        )
    )
    
    return fig


def plot_risk_distribution(
    predictions: List[Dict[str, Any]],
    title: str = "Risk Level Distribution"
) -> go.Figure:
    """
    Create an interactive pie chart showing the distribution of risk categories.
    
    Visualizes how many bookings fall into each risk category (low, medium, high)
    with color coding and percentage labels.
    
    Args:
        predictions: List of prediction dictionaries, each containing 'risk_level' key
        title: Title for the chart (default: "Risk Level Distribution")
    
    Returns:
        go.Figure: Plotly figure object with pie chart
    
    Example:
        >>> predictions = [
        ...     {'risk_level': 'low', 'probability': 0.2},
        ...     {'risk_level': 'high', 'probability': 0.8},
        ...     {'risk_level': 'medium', 'probability': 0.5}
        ... ]
        >>> fig = plot_risk_distribution(predictions)
        >>> fig.show()
    """
    # Count risk levels
    risk_counts = {
        'low': 0,
        'medium': 0,
        'high': 0
    }
    
    for pred in predictions:
        risk_level = pred.get('risk_level', 'unknown').lower()
        if risk_level in risk_counts:
            risk_counts[risk_level] += 1
    
    # Prepare data for pie chart
    labels = ['Low Risk', 'Medium Risk', 'High Risk']
    values = [risk_counts['low'], risk_counts['medium'], risk_counts['high']]
    colors = ['#00cc66', '#ffa500', '#ff4b4b']  # Green, Orange, Red
    
    # Create pie chart
    fig = go.Figure(data=[
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(
                colors=colors,
                line=dict(color='white', width=2)
            ),
            textinfo='label+percent+value',
            textfont=dict(size=14),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>',
            hole=0.3  # Create a donut chart
        )
    ])
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        height=400,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="white",
        font={'family': "Arial, sans-serif"},
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        annotations=[
            dict(
                text=f'Total<br>{sum(values)}',
                x=0.5,
                y=0.5,
                font_size=16,
                showarrow=False
            )
        ]
    )
    
    return fig


def plot_probability_distribution(
    predictions: List[Dict[str, Any]],
    title: str = "Cancellation Probability Distribution",
    bins: int = 20
) -> go.Figure:
    """
    Create a histogram showing the distribution of cancellation probabilities.
    
    Useful for batch predictions to visualize the overall distribution of
    cancellation probabilities across multiple bookings.
    
    Args:
        predictions: List of prediction dictionaries, each containing 'probability' key
        title: Title for the chart (default: "Cancellation Probability Distribution")
        bins: Number of bins for the histogram (default: 20)
    
    Returns:
        go.Figure: Plotly figure object with histogram
    
    Example:
        >>> predictions = [
        ...     {'probability': 0.2},
        ...     {'probability': 0.8},
        ...     {'probability': 0.5}
        ... ]
        >>> fig = plot_probability_distribution(predictions)
        >>> fig.show()
    """
    # Extract probabilities
    probabilities = [pred.get('probability', 0) * 100 for pred in predictions]
    
    # Create histogram
    fig = go.Figure(data=[
        go.Histogram(
            x=probabilities,
            nbinsx=bins,
            marker=dict(
                color='#1f77b4',
                line=dict(color='white', width=1)
            ),
            hovertemplate='Probability Range: %{x}<br>Count: %{y}<extra></extra>'
        )
    ])
    
    # Add vertical lines for risk thresholds
    fig.add_vline(
        x=30,
        line_dash="dash",
        line_color="green",
        annotation_text="Low/Medium Threshold",
        annotation_position="top"
    )
    
    fig.add_vline(
        x=70,
        line_dash="dash",
        line_color="red",
        annotation_text="Medium/High Threshold",
        annotation_position="top"
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Cancellation Probability (%)",
        yaxis_title="Number of Bookings",
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        paper_bgcolor="white",
        plot_bgcolor="rgba(240, 240, 240, 0.5)",
        font={'family': "Arial, sans-serif"},
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            range=[0, 100]
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        bargap=0.1
    )
    
    return fig


def plot_feature_contributions(
    booking_data: Dict[str, Any],
    feature_importance: Dict[str, float],
    top_n: int = 10,
    title: str = "Feature Contributions to This Prediction"
) -> go.Figure:
    """
    Create a bar chart showing how specific booking features contribute to the prediction.
    
    This visualization helps explain why a particular prediction was made by showing
    the values of the most important features for this specific booking.
    
    Args:
        booking_data: Dictionary containing the booking features and their values
        feature_importance: Dictionary mapping feature names to importance scores
        top_n: Number of top features to display (default: 10)
        title: Title for the chart (default: "Feature Contributions to This Prediction")
    
    Returns:
        go.Figure: Plotly figure object with grouped bar chart
    
    Example:
        >>> booking = {'lead_time': 342, 'adr': 95.5, 'deposit_type': 'No Deposit'}
        >>> importance = {'lead_time': 0.25, 'adr': 0.20, 'deposit_type': 0.15}
        >>> fig = plot_feature_contributions(booking, importance)
        >>> fig.show()
    """
    # Get top N most important features
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    feature_names = [f[0] for f in sorted_features]
    importance_values = [f[1] for f in sorted_features]
    
    # Get feature values from booking data (normalize for display)
    feature_values = []
    for feature in feature_names:
        value = booking_data.get(feature, 'N/A')
        feature_values.append(str(value)[:20])  # Truncate long values
    
    # Reverse for better visualization
    feature_names = feature_names[::-1]
    importance_values = importance_values[::-1]
    feature_values = feature_values[::-1]
    
    # Create horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=importance_values,
            y=feature_names,
            orientation='h',
            marker=dict(
                color='#1f77b4',
                line=dict(color='rgb(8,48,107)', width=1.5)
            ),
            text=feature_values,
            textposition='outside',
            textfont=dict(size=10),
            hovertemplate='<b>%{y}</b><br>Value: %{text}<br>Importance: %{x:.4f}<extra></extra>'
        )
    ])
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="Feature Importance",
        yaxis_title="Feature (with current value)",
        height=max(400, top_n * 40),
        margin=dict(l=150, r=100, t=80, b=50),
        paper_bgcolor="white",
        plot_bgcolor="rgba(240, 240, 240, 0.5)",
        font={'family': "Arial, sans-serif"},
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=False
        )
    )
    
    return fig


def create_summary_metrics_chart(
    predictions: List[Dict[str, Any]],
    title: str = "Batch Prediction Summary"
) -> go.Figure:
    """
    Create a summary chart with key metrics for batch predictions.
    
    Displays multiple metrics in a single visualization including total bookings,
    predicted cancellations, average probability, and risk distribution.
    
    Args:
        predictions: List of prediction dictionaries
        title: Title for the chart (default: "Batch Prediction Summary")
    
    Returns:
        go.Figure: Plotly figure object with indicator chart
    
    Example:
        >>> predictions = [
        ...     {'prediction': 1, 'probability': 0.8, 'risk_level': 'high'},
        ...     {'prediction': 0, 'probability': 0.2, 'risk_level': 'low'}
        ... ]
        >>> fig = create_summary_metrics_chart(predictions)
        >>> fig.show()
    """
    # Calculate metrics
    total_bookings = len(predictions)
    predicted_cancellations = sum(1 for p in predictions if p.get('prediction') == 1)
    cancellation_rate = (predicted_cancellations / total_bookings * 100) if total_bookings > 0 else 0
    
    valid_probabilities = [p.get('probability', 0) for p in predictions if p.get('probability') is not None]
    avg_probability = (sum(valid_probabilities) / len(valid_probabilities) * 100) if valid_probabilities else 0
    
    high_risk_count = sum(1 for p in predictions if p.get('risk_level') == 'high')
    high_risk_rate = (high_risk_count / total_bookings * 100) if total_bookings > 0 else 0
    
    # Create indicator chart with multiple metrics
    fig = go.Figure()
    
    # Add indicators
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=total_bookings,
        title={'text': "Total Bookings"},
        domain={'x': [0, 0.25], 'y': [0.5, 1]}
    ))
    
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=predicted_cancellations,
        title={'text': "Predicted Cancellations"},
        delta={'reference': total_bookings / 2, 'relative': False},
        domain={'x': [0.25, 0.5], 'y': [0.5, 1]}
    ))
    
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=cancellation_rate,
        title={'text': "Cancellation Rate (%)"},
        number={'suffix': "%"},
        domain={'x': [0.5, 0.75], 'y': [0.5, 1]}
    ))
    
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=high_risk_count,
        title={'text': "High Risk Bookings"},
        domain={'x': [0.75, 1], 'y': [0.5, 1]}
    ))
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=avg_probability,
        title={'text': "Avg Cancellation Probability"},
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, 30], 'color': '#e6f7ed'},
                {'range': [30, 70], 'color': '#fff4e6'},
                {'range': [70, 100], 'color': '#ffe6e6'}
            ]
        },
        domain={'x': [0.25, 0.75], 'y': [0, 0.4]}
    ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=500,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="white",
        font={'family': "Arial, sans-serif"}
    )
    
    return fig
