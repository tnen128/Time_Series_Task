import pandas as pd
from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import numpy as np
import joblib
import os

app = Flask(__name__)

def preprocess_data(data):
    df = pd.DataFrame(data['values'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['value'] = df['value'].interpolate(method='linear')
    df.set_index('timestamp', inplace=True)
    
    for lag in range(1, 4):
        df[f'lag_{lag}'] = df['value'].shift(lag)
    
    df.fillna(method='bfill', inplace=True)
    
    df['rolling_mean'] = df['value'].rolling(window=3).mean()
    df['rolling_std'] = df['value'].rolling(window=3).std()
    
    fft_components = np.fft.fft(df['value'])
    df['fft_real'] = np.real(fft_components)
    
    df['minute_of_hour'] = df.index.minute
    
    feature_cols = ['lag_1', 'lag_2', 'lag_3', 'rolling_mean', 'rolling_std', 'fft_real', 'minute_of_hour']
    
    df_filled = df[feature_cols].fillna(method='bfill')
    
    return df_filled.tail(1)

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
    
    df_processed = preprocess_data(data)
    
    model = load_model(dataset_id)
    if model is None:
        return jsonify({"error": "Model not found for dataset_id"}), 404
    
    feature_cols = ['lag_1', 'lag_2', 'lag_3', 'rolling_mean', 'rolling_std', 'fft_real', 'minute_of_hour']
    
    if not all(col in df_processed.columns for col in feature_cols):
        return jsonify({"error": "Missing features in the processed data"}), 400
    
    features = df_processed[feature_cols].values
    
    try:
        prediction = model.predict(features)[0]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
