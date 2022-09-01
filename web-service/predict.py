import os
import pickle

import mlflow
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify

BUCKET_NAME = os.getenv('BUCKET_NAME', 'mlflow-models-bryskulov')
MLFLOW_EXPERIMENT_ID = os.getenv('MLFLOW_EXPERIMENT_ID', '1')
RUN_ID = os.getenv('RUN_ID', 'b94644a7545e431781807a3001f97c14')


def load_model():
    logged_model = f's3://{BUCKET_NAME}/{MLFLOW_EXPERIMENT_ID}/{RUN_ID}/artifacts/models_mlflow'
    model = mlflow.pyfunc.load_model(logged_model)

    logged_dv = f's3://{BUCKET_NAME}/{MLFLOW_EXPERIMENT_ID}/{RUN_ID}/artifacts/preprocessor/preprocessor.b'
    dv_location = mlflow.artifacts.download_artifacts(logged_dv)
    with open(dv_location, 'rb') as f_in:
        dv = pickle.load(f_in)
    
    return model, dv

def prepare_features(house):
    features = house
    features['TotalSF'] = house['TotalBsmtSF'] + house['1stFlrSF'] + house['2ndFlrSF']
    
    return features

def predict(features, model, dv):
    X = dv.transform(features)
    preds = model.predict(X)
    return float(preds[0])


app = Flask('price-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    house = request.get_json()

    features = prepare_features(house)
    model, dv = load_model()
    pred = predict(features, model, dv)

    result = {
        'price': pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

