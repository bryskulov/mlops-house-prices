import json
import pickle
from pathlib import Path

import mlflow
from sklearn.feature_extraction import DictVectorizer

import predict


def read_json(file):
    test_directory = Path(__file__).parent

    with open(test_directory / file, encoding='utf-8') as json_file:
        return json.load(json_file)


def test_prepare_features():
    data_input = read_json('house_data.json')
    actual_result = predict.prepare_features(data_input)['TotalSF']
    expected_result = 2524

    assert actual_result == expected_result


def test_load_model():
    actual_result_model, actual_result_dv = predict.load_model()
    print(type(actual_result_model))
    assert isinstance(actual_result_model, mlflow.pyfunc.PyFuncModel)
    assert isinstance(actual_result_dv, DictVectorizer)


def test_predict():
    data_input = read_json('house_data.json')
    features = predict.prepare_features(data_input)

    test_directory = Path(__file__).parent
    with open(test_directory / 'mock_models/model.xgb', 'rb') as f_in:
        model = pickle.load(f_in)

    with open(test_directory / 'mock_models/preprocessor.b', 'rb') as f_in:
        dv = pickle.load(f_in)

    actual_result = int(predict.predict(features, model, dv))

    expected_result = 173729

    assert actual_result == expected_result
