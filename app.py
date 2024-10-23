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
    
    # Separate the last value to avoid it being included in interpolation
    last_value = df.iloc[-1]['value']
    df.loc[df.index[-1], 'value'] = np.nan
    
    # Perform interpolation on the 'value' column
    df['value'] = df['value'].interpolate(method='linear')
    
    # Restore the last value after interpolation
    df.loc[df.index[-1], 'value'] = last_value
    
    df.set_index('timestamp', inplace=True)
    
    # Generate lag features
    for lag in range(1, 4):
        df[f'lag_{lag}'] = df['value'].shift(lag)
    
    # Generate rolling statistics features
    df['rolling_mean'] = df['value'].shift(1).rolling(window=3).mean()
    df['rolling_std'] = df['value'].shift(1).rolling(window=3).std()

 
    # Add the minute of the hour as a feature
    df['minute_of_hour'] = df.index.minute
  
    return df.tail(1)

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
    
    feature_cols = [ 'minute_of_hour','lag_1', 'lag_2', 'lag_3', 'rolling_mean', 'rolling_std']
    
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
