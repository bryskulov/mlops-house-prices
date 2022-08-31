import os
import pickle

import mlflow
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify

RUN_ID = os.getenv('RUN_ID')
BUCKET_NAME = os.getenv('BUCKET_NAME')

logged_model = f's3://{BUCKET_NAME}/1/{RUN_ID}/artifacts/models_mlflow'
model = mlflow.pyfunc.load_model(logged_model)

logged_dv = f's3://{BUCKET_NAME}/1/{RUN_ID}/artifacts/preprocessor/preprocessor.b'
dv_location = mlflow.artifacts.download_artifacts(logged_dv)
with open(dv_location, 'rb') as f_in:
    dv = pickle.load(f_in)


def prepare_features(house):
    # df = pd.DataFrame.from_dict(ride, orient='index')
    # df.reset_index(level=0, inplace=True)

    # num_to_categ_cols = ['MSSubClass', 'OverallCond', 'YrSold', 'MoSold']
    # df[num_to_categ_cols] = df[num_to_categ_cols].astype(str)
    features = house
    features['TotalSF'] = house['TotalBsmtSF'] + house['1stFlrSF'] + house['2ndFlrSF']
    
    return features

def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return float(preds[0])


app = Flask('price-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    house = request.get_json()

    features = prepare_features(house)
    pred = predict(features)

    result = {
        'price': pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

