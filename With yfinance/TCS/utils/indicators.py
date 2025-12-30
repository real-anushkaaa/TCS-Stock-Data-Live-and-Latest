import pandas as pd
import numpy as np

def calculate_buy_sell_signals(df):
    """
    Calculate buy/sell signals based on EMA crossover strategy
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing price data with technical indicators
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with buy/sell signals
    """
    # Make a copy to avoid modifying the original dataframe
    signals_df = df.copy()
    
    # Calculate short and long EMAs if not already present
    if 'EMA9' not in signals_df.columns:
        signals_df['EMA9'] = signals_df['Close'].ewm(span=9, adjust=False).mean()
    
    if 'EMA21' not in signals_df.columns:
        signals_df['EMA21'] = signals_df['Close'].ewm(span=21, adjust=False).mean()
    
    # Initialize signal column
    signals_df['Signal'] = 0
    
    # Generate buy/sell signals based on EMA crossover
    # Buy signal (1) when short EMA crosses above long EMA
    # Sell signal (-1) when short EMA crosses below long EMA
    signals_df['Signal'] = np.where(signals_df['EMA9'] > signals_df['EMA21'], 1, 0)
    signals_df['Position_Change'] = signals_df['Signal'].diff()
    
    # Filter only the rows where position changes
    buy_signals = signals_df[signals_df['Position_Change'] == 1].copy()
    sell_signals = signals_df[signals_df['Position_Change'] == -1].copy()
    
    # Create a combined signals dataframe
    if not buy_signals.empty or not sell_signals.empty:
        # Add signal type
        if not buy_signals.empty:
            buy_signals['Signal_Type'] = 'BUY'
        if not sell_signals.empty:
            sell_signals['Signal_Type'] = 'SELL'
        
        # Combine buy and sell signals
        combined_signals = pd.concat([buy_signals, sell_signals])
        
        # Sort by date
        if 'Date' in combined_signals.columns:
            combined_signals = combined_signals.sort_values('Date')
        
        # Select relevant columns
        signal_columns = ['Date', 'Close', 'Signal_Type']
        return combined_signals[signal_columns]
    else:
        return pd.DataFrame()

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """
    Calculate the Sharpe ratio for a series of returns
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of returns
    risk_free_rate : float, optional
        Risk-free rate, defaults to 0.0
        
    Returns:
    --------
    float
        Sharpe ratio
    """
    excess_returns = returns - risk_free_rate
    if excess_returns.std() == 0:
        return 0
    return excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # Annualized