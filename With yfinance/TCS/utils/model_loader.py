import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model

def load_model_and_scalers(model_type, timeframe_prefix):
    """
    Load the appropriate model and scalers based on model type and timeframe
    
    Parameters:
    -----------
    model_type : str
        Type of model to load ('RandomForest', 'XGBoost', or 'LSTM')
    timeframe_prefix : str
        Prefix for the timeframe ('1d', '5d', '1m', '6m', '1y', '5y')
        
    Returns:
    --------
    tuple
        (model, X_scaler, y_scaler)
    """
    models_dir = 'models'
    
    # Map model_type to file prefix
    model_prefix_map = {
        'RandomForest': 'rf',
        'XGBoost': 'xgb',
        'LSTM': 'lstm'
    }
    
    model_prefix = model_prefix_map.get(model_type)
    
    if not model_prefix:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Construct model path
    if model_type == 'LSTM':
        model_path = os.path.join(models_dir, f"{model_prefix}_{timeframe_prefix}_finetuned.keras")
        X_scaler_path = os.path.join(models_dir, f"X_scaler_{timeframe_prefix}_finetuned.pkl")
        y_scaler_path = os.path.join(models_dir, f"y_scaler_{timeframe_prefix}_finetuned.pkl")
    else:  # RandomForest or XGBoost
        model_path = os.path.join(models_dir, f"{model_prefix}_{timeframe_prefix}.pkl")
        # For RF/XGB, we might not have separate scalers
        X_scaler_path = os.path.join(models_dir, f"X_scaler_{timeframe_prefix}.pkl")
        y_scaler_path = os.path.join(models_dir, f"y_scaler_{timeframe_prefix}.pkl")
    
    # Load model
    try:
        if model_type == 'LSTM':
            model = load_model(model_path)
        else:  # RandomForest or XGBoost
            model = joblib.load(model_path)
    except Exception as e:
        raise Exception(f"Failed to load model from {model_path}: {str(e)}")
    
    # Load X scaler if available
    try:
        if os.path.exists(X_scaler_path):
            X_scaler = joblib.load(X_scaler_path)
        else:
            X_scaler = None
    except Exception:
        X_scaler = None
    
    # Load y scaler if available
    try:
        if os.path.exists(y_scaler_path):
            y_scaler = joblib.load(y_scaler_path)
        else:
            y_scaler = None
    except Exception:
        y_scaler = None
    
    return model, X_scaler, y_scaler