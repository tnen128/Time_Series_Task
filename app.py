import pandas as pd
from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# Preprocess the data depending on the model type
def preprocess_data(data, model_name):
    df = pd.DataFrame(data['values'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    
    df.set_index('timestamp', inplace=True)

    # Handle preprocessing differently for different models
    if model_name == 'Prophet':
        # Prophet expects a DataFrame with columns 'ds' and 'y'
        df.reset_index(inplace=True)
        df.rename(columns={'timestamp': 'ds', 'value': 'y'}, inplace=True)
        return df
    
    elif model_name == 'ARIMA':
        # ARIMA requires raw time series data, return 'value' column as a series
        return df['value']
    
    else:
        
        last_value = df.iloc[-1]['value']
        df.loc[df.index[-1], 'value'] = np.nan
        
        # Interpolate missing values
        df['value'] = df['value'].interpolate(method='linear')
        df.loc[df.index[-1], 'value'] = last_value
        
        # Generate lag features
        for lag in range(1, 4):
            df[f'lag_{lag}'] = df['value'].shift(lag)
        
        # Generate rolling statistics features
        df['rolling_mean'] = df['value'].shift(1).rolling(window=3).mean()
        df['rolling_std'] = df['value'].shift(1).rolling(window=3).std()
        
        # Add time-based features
        df['minute_of_hour'] = df.index.minute
        
        return df.tail(1)  # Return the last row with features for prediction

# Load model based on the dataset_id
def load_model(dataset_id):
    model_path = f'models/model_{dataset_id}.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    dataset_id = data.get('dataset_id')
    
    # Load the model
    model = load_model(dataset_id)
    if model is None:
        return jsonify({"error": "Model not found for dataset_id"}), 404
    
    # Determine the type of model
    model_name = type(model).__name__.lower()

    # Preprocess data based on model type
    df_processed = preprocess_data(data, model_name)

    try:
        if model_name == 'prophet':
            future = model.make_future_dataframe(periods=1, freq='min')
            forecast = model.predict(future)
            prediction = forecast['yhat'].iloc[-1]

        elif model_name == 'arima':
            prediction = model.forecast(steps=1)[0]

        elif model_name == 'xgbregressor' or model_name == 'linearregression':
            # For XGBoost and LinearRegression
            feature_cols = ['minute_of_hour', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean', 'rolling_std']
            
            if not all(col in df_processed.columns for col in feature_cols):
                return jsonify({"error": "Missing features in the processed data"}), 400
            
            features = df_processed[feature_cols].values
            prediction = model.predict(features)[0]
        else:
            return jsonify({"error": f"Unsupported model type: {model_name}"}), 400

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    prediction = float(prediction)
    
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)
