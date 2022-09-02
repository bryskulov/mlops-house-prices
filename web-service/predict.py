import os
import pickle

import mlflow
from flask import Flask, request, jsonify

S3_BUCKET_PATH = os.getenv('S3_BUCKET_PATH')
MLFLOW_EXPERIMENT_ID = os.getenv('MLFLOW_EXPERIMENT_ID')
RUN_ID = os.getenv('RUN_ID')


def load_model():
    logged_model = f'{S3_BUCKET_PATH}/{MLFLOW_EXPERIMENT_ID}/{RUN_ID}/artifacts/models_mlflow'
    model = mlflow.pyfunc.load_model(logged_model)

    logged_dv = f'{S3_BUCKET_PATH}/{MLFLOW_EXPERIMENT_ID}/{RUN_ID}/artifacts/preprocessor/preprocessor.b'
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

