# House price prediction

Project for [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)
The main goal of the project is to use MLOps tools and best practices on prediction task.

Dataset source: Kaggle House Prices Prediction challange [link](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/rules)

Modelling notebook is inspired by [Serigne's notebook](https://www.kaggle.com/code/serigne/stacked-regressions-top-4-on-leaderboard)

# Problem Definition

This project tries to automate the prediction of house prices based on different features of a house such as location, shape, available utilities, condition, style, etc. The project intents to automate different stages of the process including training, deployment and further sustaining it in production.


# Documentation

## Installation

The project is developed on AWS EC2 instance and it is highly recommended to run on EC2 instance as well.
Model artifacts are stored on AWS S3 bucket, so it is advised to create a S3 bucket with you custom name.

**Programs installed on EC2**: Anaconda, Docker, docker-compose

Clone this repository to the local repository

```bash
    git clone https://github.com/bryskulov/mlops-house-prices.git
```

Folder explanations:
- notebooks: Jupyter notebooks for prototyping
- model_training: Automated model training scripts
- web_service: Deployment of the model as a web-service

## Model training (model_training/)

First, install `pipenv` package and later the other packages from `Pipfile`.
It is important to be in the same directory as the `Pipfile`, when running the bash script.

```bash
    pip install pipenv
    pipenv install
```

Activate the pipenv environment:

```bash
    pipenv shell
```

Set your AWS S3 Bucket name as environment variable:
```bash
    export S3_BUCKET_PATH="s3://mlflow-models-bryskulov"
```

### Train model once with Python CLI

This script is used to train the model once using the data in "model_training/data/" path. 
The idea is that new models are to be trained depending on the data in that folder.
In future, of course, it is better to pull data from some relational database.

Activate MLFlow Tracking server. If you need to create new database, you can use the following script

```bash
    mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root=$S3_BUCKET_PATH
```

To run the model training, run:
```bash
    python train.py --data_path data/train.csv
```


### Training with Prefect Deployment with Scheduling

Here, model training is scheduled via workflow orchestration tool "Prefect".

Activate MLFlow Tracking server:
```bash
    mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root=$S3_BUCKET_PATH
```

Start Prefect UI with the following bash command:
```bash
    prefect orion start
```
```Note```: It will run prefect server and it can accessed from the browser.


Create a new deployment with Prefect CLI command:
```bash
    prefect deployment create prefect_deploy.py 
```

```Note```: This will create a new deployment in prefect, however it won't run it.
To run the deployment we should create a work queue, it can be done in Prefect UI.

After creating work queue, we need to start the agent via bash script:

```bash
    prefect agent start <work queue ID>
```

Now, you can observe all the scheduled, completed and failed flows in the Prefect UI.

### Choosing the model

After training the models, inspect the models and choose the model that you prefer. 
Pay attention that the chosen model has an artifact attached.\

Define the chosen varibles as enrivonment variables:
```bash
    export MLFLOW_EXPERIMENT_ID='1'
    export RUN_ID='be58cd18afc44f5ab13b3409613e04f9'
```

## Deploying a model as Flask API service with MLFlow on EC2 instance (web-service/)

Don't forget to change the directory and initiate a different Pipenv environment:
```bash
    cd ..
    cd web-service
    pipenv shell
```

The web application is deployed via Flask on the localhost:9696.

To build the Docker Image run:

```bash
    docker build -t house-price-prediction-service:v2 .
```

Run the Docker:

```bash
docker run -it --rm -p 9696:9696 \
    -e S3_BUCKET_PATH=$S3_BUCKET_PATH\
    -e MLFLOW_EXPERIMENT_ID=$MLFLOW_EXPERIMENT_ID \
    -e RUN_ID=$RUN_ID \
    house-price-prediction-service:v2
```


### Testing

I run both unit tests and integration test on my deployment application.

#### Unit tests

Pytest is used for unittesting. The tests can be run through IDE or by script:

```
    pytest unit_tests
```

#### Integration test

Integration test is automated, so you only need to run script "run.sh" in the folder "integration_test":

```bash
	cd integration_test
    source run.sh
```

```Note```: If you get an error, check that you activate pipenv environment 
and passed the environment variables such as s3_bucket_path, mlflow_experiment_id, run_id