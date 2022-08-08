# House price prediction

Project for [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)
The main goal of the project is to use MLOps tools and best practices on prediction task.

Dataset source: Kaggle House Prices Prediction challange [link](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/rules)

Modelling notebook is inspired by [Serigne's notebook](https://www.kaggle.com/code/serigne/stacked-regressions-top-4-on-leaderboard)


## Documentation

### Running with Python CLI

First, clone this repository to the local repository

```bash
    git clone https://github.com/bryskulov/mlops-house-prices.git
```

First, install `pipenv` package and later the other packages from `Pipfile`.\
It is important to be in the same directory as the `Pipfile`, when running the bash script.

```bash
    pip install pipenv
    pipenv install
```

Activate the pipenv environment:

```bash
    pipenv shell
```

Activate MLFlow Tracking server. If you need to create new database, you can use the following script

```bash
    mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

To run the model training, run:

```bash
    python train.py --data_path data/train.csv
```

### Running with Prefect Deployment

Start Prefect UI with the following bash command. 
It will run prefect server and it can accessed from the browser.

```bash
    prefect orion start
```

Create a new deployment with Prefect CLI command:

```bash
    prefect deployment create prefect_deploy.py 
```

This will create a new deployment in prefect, however it won't run it.\
To run the deployment we should create a work queue, it can be done in Prefect UI.\
After creating work queue, we need to start the agent via bash script:

```bash
    prefect agent start <work queue ID>
```

Now, you can observe all the scheduled, completed and failed flows in the Prefect UI.

