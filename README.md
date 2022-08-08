# House price prediction

Project for [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)
The main goal of the project is to use MLOps tools and best practices on prediction task.

Dataset source: Kaggle House Prices Prediction challange [link](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/rules)

Modelling notebook is inspired by [Serigne's notebook](https://www.kaggle.com/code/serigne/stacked-regressions-top-4-on-leaderboard)


## Documentation

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