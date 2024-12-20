
# Time Series Model Selection and Prediction API

This repository contains a comprehensive project that involves analyzing 96 time series datasets and selecting the best model for each dataset. The project utilizes various machine learning algorithms to analyze and forecast time series data, taking into consideration different features like lag variables, rolling statistics, and Fourier components. After model selection, the project includes an API (built using Flask and Python) to make predictions using the best-fit models for each dataset.

## Project Structure

- **Notebooks**: Detailed exploratory data analysis (EDA) and model selection for 96 time series datasets.
- **API**: A Flask API that allows users to test the models by providing their own data and receiving predictions.

## Features

1. **Time Series Analysis**: Each dataset undergoes preprocessing, including:
   - Timestamp parsing and interpolation
   - Lag feature creation (`lag_1`, `lag_2`, `lag_3`)
   - Rolling statistics (mean and standard deviation)
   - Extracting additional time-based features such as minute of the hour

2. **Model Selection**: After detailed analysis, the best-performing model for each dataset is saved for future use.

3. **API for Prediction**: The API allows users to upload time series data and make predictions using the pre-trained models.

## How to Use the API

### Requirements

1. Python 3.13.0
2. Flask
3. scikit-learn
4. joblib
5. pandas
6. numpy

Install the required packages using:
```bash
pip install -r requirements.txt
```

### Running the API

1. Clone the repository:
```bash
git clone https://github.com/yourusername/timeseries-api.git](https://github.com/tnen128/Time_Series_Task.git
cd timeseries-api
```

2. Start the Flask server:
```bash
python app.py
```

By default, the Flask server will run on `http://127.0.0.1:5000/`.

### Endpoints

- **POST `/predict`**: Predicts the value for a specific time series dataset using the appropriate model.

### Input Format

To test the API using a tool like **Postman**, send a POST request to the `/predict` endpoint with the following JSON structure (4 data points at least the last one which will be predicted we extract the minutes feature but we ignore the its value):

```json
{
  "dataset_id": 9,
  "values": [
    {"timestamp": "7/1/2021  12:00:00 AM", "value": -0.925346520644351},
    {"timestamp": "7/1/2021  12:10:00 AM", "value": -1.0536322432624},
    {"timestamp": "7/1/2021  12:20:00 AM", "value": -1.02358121522349},
    {"timestamp": "7/1/2021  12:30:00 AM", "value": -0.95951936715285}
  ]
}
```
or using NULLs

```json
{
  "dataset_id": 9,
  "values": [
    {"timestamp": "7/1/2021  12:00:00 AM", "value": -0.925346520644351},
    {"timestamp": "7/1/2021  12:10:00 AM", "value": null},
    {"timestamp": "7/1/2021  12:20:00 AM", "value": -1.02358121522349},
    {"timestamp": "7/1/2021  12:30:00 AM", "value": null}

  ]
}



```

- **dataset_id**: Integer representing the ID of the dataset.
- **values**: A list of timestamp-value pairs representing the time series data.

### Response Format

A successful response will return a JSON object with the predicted value:
```json
{
    "prediction": -0.9253430128472478
}
```

### Error Handling

- If the dataset ID does not correspond to a valid model, the API will return:
```json
{
  "error": "Model not found for dataset_id"
}
```

- If there is an issue with the prediction process, the API will return:
```json
{
  "error": "Prediction failed: <error_message>"
}
```

### Testing the API Using Postman

1. Open Postman.
2. Create a new POST request.
3. Enter the following URL: `http://127.0.0.1:5000/predict`.
4. In the "Body" tab, select "raw" and choose "JSON" as the format.
5. Paste your input data as shown above.
6. Click "Send" to receive the prediction.
