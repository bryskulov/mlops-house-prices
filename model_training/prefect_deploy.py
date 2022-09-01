import pickle

import mlflow
import numpy as np
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from prefect import task, flow


RANDOM_SEED=42

@task
def read_data(filename):
    df = pd.read_csv(filename)
    
    df = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<300000)].index)
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    
    return df

@task
def prepare_features(df: pd.DataFrame):
    
    df_label = df.SalePrice.values
    df.drop(['SalePrice'], axis=1, inplace=True) 
    
    num_to_categ_cols = ['MSSubClass', 'OverallCond', 'YrSold', 'MoSold']
    df[num_to_categ_cols] = df[num_to_categ_cols].astype(str)
    
    dv = DictVectorizer(sparse=True)
    df_dict = df.to_dict("records")
    df_processed = dv.fit_transform(df_dict)
    
    return df_processed, df_label, dv

@task
def split_dataset(X, y, split_sizes=[0.8, 0.5], random_seed=42):
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, 
        train_size=split_sizes[0], 
        random_state=random_seed)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, 
        test_size=split_sizes[1], 
        random_state=random_seed)
    return X_train, y_train, X_valid, y_valid, X_test, y_test

@task
def train_model_search(train, valid, test, y_test):
    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)

            model_xgb = xgb.train(
                params=params,
                dtrain=train,
                evals=[(valid, 'validation')],
                num_boost_round=2200,
                early_stopping_rounds=50)

            y_pred = model_xgb.predict(test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)
        mlflow.end_run()

        return {'loss': rmse, 'status': STATUS_OK}
    
    search_space = {
        'gamma': hp.loguniform('gamma', -5, -1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.3,1),
        'subsample': hp.uniform('subsample', 0.4,1),
        'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:linear',
        'seed': RANDOM_SEED
    }
    
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=Trials()
    )

@task
def train_best_model(train, valid, test, y_test, dv):
    with mlflow.start_run():

        best_params = {
            'colsample_bytree': 0.9250870893919794,
            'gamma': 0.007995628667745471,
            'learning_rate': 0.20384373996439606,
            'max_depth': 4,
            'min_child_weight': 0.41092408055939844,
            'reg_alpha': 0.007444391334457018,
            'reg_lambda': 0.017392816466180783,
            'subsample': 0.7772671896767146,
            'n_estimators': 2200,
            'seed': RANDOM_SEED
        }

        mlflow.set_tag("type", "best_model")
        mlflow.log_params(best_params)

        model_xgb = xgb.train(
            params=best_params,
            dtrain=train,
            evals=[(valid, 'validation')],
            num_boost_round=2200,
            early_stopping_rounds=50)

        y_pred = model_xgb.predict(test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(model_xgb, artifact_path="models_mlflow")
    mlflow.end_run()


@flow
def main(data_path: str = "./data/train.csv"):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("house-price-prediction")

    df = read_data(data_path)
    X, y, dv = prepare_features(df).result()
    X_train, y_train, X_valid, y_valid, X_test, y_test = split_dataset(X, y).result()

    train_xgb = xgb.DMatrix(X_train, label=y_train)
    valid_xgb = xgb.DMatrix(X_valid, label=y_valid)
    test_xgb = xgb.DMatrix(X_test, label=y_test)

    train_model_search(train_xgb, valid_xgb, test_xgb, y_test)
    train_best_model(train_xgb, valid_xgb, test_xgb, y_test, dv)   


from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

DeploymentSpec(
  flow=main,
  name="model_training_prefect",
  schedule=IntervalSchedule(interval=timedelta(minutes=30)),
  flow_runner=SubprocessFlowRunner(),
  tags=["ml"]
)