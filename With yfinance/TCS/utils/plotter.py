import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

def plot_candlestick(df, pred_dates=None, pred_values=None):
    """
    Create a candlestick chart with optional prediction overlay
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing OHLC data
    pred_dates : pandas.Series or array-like, optional
        Dates for predictions
    pred_values : pandas.Series or array-like, optional
        Predicted values
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='TCS Price'
    ))
    
    # Add MA20 and MA50 if available
    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['MA20'],
            line=dict(color='blue', width=1),
            name='MA20'
        ))
    
    if 'MA50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['MA50'],
            line=dict(color='orange', width=1),
            name='MA50'
        ))
    
    # Add predictions if available
    if pred_dates is not None and pred_values is not None:
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=pred_values,
            line=dict(color='red', width=2, dash='dash'),
            name='Predicted'
        ))
    
    # Update layout
    fig.update_layout(
        title='TCS Stock Price with Predictions',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    return fig

def plot_predictions(dates, actual, predicted):
    """
    Create a line chart comparing actual vs predicted values
    
    Parameters:
    -----------
    dates : pandas.Series or array-like
        Dates for the data points
    actual : pandas.Series or array-like
        Actual values
    predicted : pandas.Series or array-like
        Predicted values
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    # Add actual values
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))
    
    # Add predicted values
    fig.add_trace(go.Scatter(
        x=dates,
        y=predicted,
        mode='lines',
        name='Predicted',
        line=dict(color='red', dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='Actual vs Predicted Prices',
        xaxis_title='Date',
        yaxis_title='Price',
        height=400
    )
    
    return fig

def plot_indicators(df):
    """
    Create a subplot with technical indicators
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing price data and technical indicators
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Create subplots: 3 rows, 1 column
    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=('Price', 'RSI', 'MACD'),
                        row_heights=[0.5, 0.25, 0.25])
    
    # Add price chart to first row
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='white')
    ), row=1, col=1)
    
    # Add RSI to second row if available
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple')
        ), row=2, col=1)
        
        # Add RSI overbought/oversold lines
        fig.add_trace(go.Scatter(
            x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
            y=[70, 70],
            mode='lines',
            line=dict(color='red', width=1, dash='dash'),
            showlegend=False
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=[df['Date'].iloc[0], df['Date'].iloc[-1]],
            y=[30, 30],
            mode='lines',
            line=dict(color='green', width=1, dash='dash'),
            showlegend=False
        ), row=2, col=1)
    
    # Add MACD to third row if available
    if 'MACD' in df.columns and 'Signal_Line' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue')
        ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Signal_Line'],
            mode='lines',
            name='Signal Line',
            line=dict(color='red')
        ), row=3, col=1)
        
        # Calculate and add MACD histogram
        if 'MACD' in df.columns and 'Signal_Line' in df.columns:
            macd_hist = df['MACD'] - df['Signal_Line']
            colors = ['green' if val >= 0 else 'red' for val in macd_hist]
            
            fig.add_trace(go.Bar(
                x=df['Date'],
                y=macd_hist,
                name='MACD Histogram',
                marker_color=colors,
                showlegend=False
            ), row=3, col=1)
    
    # Update layout
    fig.update_layout(
        title='Technical Indicators',
        height=800,
        xaxis3_title='Date',
        yaxis_title='Price',
        yaxis2_title='RSI',
        yaxis3_title='MACD'
    )
    
    return fig

def plot_macro_indicators(end_date, lookback_days):
    """
    Create plots for macroeconomic indicators
    
    Parameters:
    -----------
    end_date : datetime
        End date for the data
    lookback_days : int
        Number of days to look back
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Calculate start date
    start_date = end_date - timedelta(days=lookback_days)
    
    # Load macro indicator data
    try:
        nifty50_df = pd.read_csv('data/nifty50.csv')
        usdinr_df = pd.read_csv('data/usdinr.csv')
        
        # Convert dates
        nifty50_df['Date'] = pd.to_datetime(nifty50_df['Date'])
        usdinr_df['Date'] = pd.to_datetime(usdinr_df['Date'])
        
        # Filter by date range
        nifty50_df = nifty50_df[(nifty50_df['Date'] >= start_date) & (nifty50_df['Date'] <= end_date)]
        usdinr_df = usdinr_df[(usdinr_df['Date'] >= start_date) & (usdinr_df['Date'] <= end_date)]
        
        # Create subplots: 2 rows, 1 column
        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.1,
                            subplot_titles=('NIFTY 50', 'USD/INR'),
                            row_heights=[0.5, 0.5])
        
        # Add NIFTY 50 to first row
        fig.add_trace(go.Scatter(
            x=nifty50_df['Date'],
            y=nifty50_df['Close'],
            mode='lines',
            name='NIFTY 50',
            line=dict(color='blue')
        ), row=1, col=1)
        
        # Add USD/INR to second row
        fig.add_trace(go.Scatter(
            x=usdinr_df['Date'],
            y=usdinr_df['Close'],
            mode='lines',
            name='USD/INR',
            line=dict(color='green')
        ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title='Macroeconomic Indicators',
            height=600,
            xaxis2_title='Date',
            yaxis_title='NIFTY 50',
            yaxis2_title='USD/INR'
        )
        
        return fig
    
    except Exception as e:
        # If data loading fails, create an empty figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading macro indicator data: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(height=600)
        return fig