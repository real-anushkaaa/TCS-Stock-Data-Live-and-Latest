import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
from utils.model_loader import load_model_and_scalers
from utils.indicators import calculate_buy_sell_signals
from utils.plotter import plot_candlestick, plot_predictions, plot_indicators, plot_macro_indicators
from utils.sentiment import analyze_news_sentiment

# Try to import pandas_ta, but don't fail if it's not available
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

# Page configuration
st.set_page_config(page_title="TCS Stock Price Prediction", layout="wide")

# App title and description
st.title("TCS Stock Price Prediction Dashboard")
st.markdown("""
This app predicts TCS stock prices using machine learning models trained on historical data.
Select a timeframe and model type from the sidebar to view predictions and analysis.
""")

# Sidebar for controls
st.sidebar.title("Controls")

# Timeframe selector
TIMEFRAME_MODEL_MAP = {
    "1D (1-min)": ["RandomForest", "XGBoost"],
    "5D (30-min)": ["RandomForest", "XGBoost"],
    "1M (daily)": ["RandomForest", "XGBoost"],
    "6M (daily)": ["RandomForest", "XGBoost"],
    "1Y (daily)": ["LSTM"],
    "5Y (weekly)": ["LSTM"]
}

timeframe = st.sidebar.selectbox(
    "Select Timeframe",
    list(TIMEFRAME_MODEL_MAP.keys())
)

# Model selector (restricted by timeframe)
model_type = st.sidebar.radio(
    "Select Model Type",
    TIMEFRAME_MODEL_MAP[timeframe]
)

# Date picker
end_date = st.sidebar.date_input(
    "Select End Date",
    datetime.now().date()
)

# Predict button
predict_button = st.sidebar.button("Predict")

# Map timeframe to file and model names
timeframe_map = {
    "1D (1-min)": {"file": "tcs_1d_features.csv", "model_prefix": "1d", "interval": "1-min", "lookback": 7},
    "5D (30-min)": {"file": "tcs_5d_features.csv", "model_prefix": "5d", "interval": "30-min", "lookback": 20},
    "1M (daily)": {"file": "tcs_1m_features.csv", "model_prefix": "1m", "interval": "daily", "lookback": 90},
    "6M (daily)": {"file": "tcs_6m_features.csv", "model_prefix": "6m", "interval": "daily", "lookback": 180},
    "1Y (daily)": {"file": "tcs_1y_features.csv", "model_prefix": "1y", "interval": "daily", "lookback": 365},
    "5Y (weekly)": {"file": "tcs_5y_features.csv", "model_prefix": "5y", "interval": "weekly", "lookback": 260}
}

# Main function
def main():
    if predict_button:
        # Get timeframe details
        timeframe_details = timeframe_map[timeframe]
        file_name = timeframe_details["file"]
        model_prefix = timeframe_details["model_prefix"]
        interval = timeframe_details["interval"]
        
        # Load data
        try:
            data_path = f"data/processed/{file_name}"
            df = pd.read_csv(data_path)
            if 'Date' not in df.columns:
                st.error("The loaded data does not contain a 'Date' column. Please check your data file.")
                return
            st.success(f"Loaded data for {timeframe}")
            
            # Filter data based on end date if applicable
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                end_date_dt = pd.to_datetime(end_date)
                df = df[df['Date'] <= end_date_dt]
            
            # Ensure model_type is always set before use
            selected_model_type = model_type
            selected_model_prefix = model_prefix
            fallback_used = False
            # Load appropriate model based on timeframe and model type
            if selected_model_type == "LSTM" and (selected_model_prefix == "1y" or selected_model_prefix == "5y"):
                model, X_scaler, y_scaler = load_model_and_scalers(selected_model_type, selected_model_prefix)
                st.success(f"Loaded {selected_model_type} model for {timeframe}")
            elif selected_model_type in ["RandomForest", "XGBoost"] and selected_model_prefix in ["1d", "5d", "1m", "6m"]:
                model, X_scaler, y_scaler = load_model_and_scalers(selected_model_type, selected_model_prefix)
                st.success(f"Loaded {selected_model_type} model for {timeframe}")
            else:
                st.warning(f"{selected_model_type} model not available for {timeframe}. Using fallback model.")
                fallback_used = True
                if selected_model_prefix in ["1y", "5y"]:
                    selected_model_type = "LSTM"
                else:
                    selected_model_type = "RandomForest"
                try:
                    model, X_scaler, y_scaler = load_model_and_scalers(selected_model_type, selected_model_prefix)
                    st.info(f"Using {selected_model_type} model instead")
                except Exception as fallback_e:
                    st.error(f"Fallback model loading failed: {str(fallback_e)}")
                    return
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["Pri]e Prediction", "Technical Indicators", "News & Sentiment", "Macro Indicators"])
            
            with tab1:
                # Remove all tabs except Price Prediction
                st.subheader("Price Prediction")
                # Make predictions
                # Dynamically align features with those present in the dataframe and used by the model
                all_possible_features = [
                    'Open', 'High', 'Low', 'Volume',
                    'MA20', 'MA50', 'RSI', 'MACD', 'Signal_Line',
                    'Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5', 'Lag_6', 'Lag_7'
                ]
                # Only use features that the model was trained on (model.feature_names_in_ if available)
                if hasattr(model, 'feature_names_in_'):
                    features = [f for f in model.feature_names_in_ if f in df.columns]
                else:
                    features = [f for f in all_possible_features if f in df.columns]
                if selected_model_type == "LSTM":
                    lookback = 180 if selected_model_prefix == "1y" else 104
                    X_scaled = X_scaler.transform(df[features].fillna(0))
                    X_seq = []
                    for i in range(lookback, len(X_scaled)):
                        X_seq.append(X_scaled[i-lookback:i])
                    X_seq = np.array(X_seq)
                    y_pred_scaled = model.predict(X_seq)
                    y_pred = y_scaler.inverse_transform(y_pred_scaled)
                    pred_dates = df['Date'].iloc[lookback:].reset_index(drop=True)
                    actual_values = df['Close'].iloc[lookback:].reset_index(drop=True)
                    pred_values = pd.Series(y_pred.flatten(), index=range(len(y_pred)))
                else:
                    X = df[features].fillna(0)
                    X_scaled = X_scaler.transform(X) if X_scaler else X
                    y_pred = model.predict(X_scaled)
                    pred_dates = df['Date']
                    actual_values = df['Close']
                    pred_values = pd.Series(y_pred, index=range(len(y_pred)))
                mae = np.mean(np.abs(actual_values - pred_values))
                rmse = np.sqrt(np.mean((actual_values - pred_values) ** 2))
                if len(actual_values) > 1:
                    ss_total = np.sum((actual_values - np.mean(actual_values)) ** 2)
                    ss_residual = np.sum((actual_values - pred_values) ** 2)
                    r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
                else:
                    r2 = 0
                returns = pred_values.pct_change().dropna()
                sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MAE", f"{mae:.2f}")
                col2.metric("RMSE", f"{rmse:.2f}")
                col3.metric("R²", f"{r2:.2f}")
                col4.metric("Sharpe Ratio", f"{sharpe:.2f}")
                fig = plot_candlestick(df, pred_dates, pred_values)
                st.plotly_chart(fig, use_container_width=True)
                fig2 = plot_predictions(pred_dates, actual_values, pred_values)
                st.plotly_chart(fig2, use_container_width=True)
                signals_df = calculate_buy_sell_signals(df)
                if not signals_df.empty:
                    st.subheader("Buy/Sell Signals (EMA Crossover)")
                    st.dataframe(signals_df)
            
            with tab2:
                st.subheader("Technical Indicators")
                
                # Plot technical indicators
                fig3 = plot_indicators(df)
                st.plotly_chart(fig3, use_container_width=True)
            
            with tab3:
                st.subheader("News & Sentiment Analysis")
                
                # Load and analyze news
                news_df = analyze_news_sentiment()
                
                # Display news with sentiment
                if not news_df.empty:
                    # Filter news by date range if applicable
                    if 'Date' in news_df.columns:
                        news_df['Date'] = pd.to_datetime(news_df['Date'])
                        start_date_dt = end_date_dt - timedelta(days=timeframe_details["lookback"])
                        filtered_news = news_df[(news_df['Date'] >= start_date_dt) & 
                                              (news_df['Date'] <= end_date_dt)]
                    else:
                        filtered_news = news_df
                    
                    if not filtered_news.empty:
                        # Display news with sentiment color coding
                        for _, row in filtered_news.iterrows():
                            sentiment = row['Sentiment']
                            if sentiment > 0.2:
                                sentiment_color = "green"
                            elif sentiment < -0.2:
                                sentiment_color = "red"
                            else:
                                sentiment_color = "gray"
                            
                            st.markdown(f"**{row['Date'].strftime('%Y-%m-%d')}**: {row['Headline']}")
                            st.markdown(f"<span style='color:{sentiment_color}'>Sentiment: {sentiment:.2f}</span>", unsafe_allow_html=True)
                            st.markdown("---")
                    else:
                        st.info("No news available for the selected date range.")
                else:
                    st.info("No news data available.")
            
            with tab4:
                st.subheader("Macroeconomic Indicators")
                
                # Load and plot macro indicators
                fig4 = plot_macro_indicators(end_date_dt, timeframe_details["lookback"])
                st.plotly_chart(fig4, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Please check if the data files and models are available in the correct directories.")

# Run the app
if __name__ == "__main__":
    main()