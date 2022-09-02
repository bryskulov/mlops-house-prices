LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")
LOCAL_IMAGE_NAME:=house-price-prediction-service:${LOCAL_TAG}

test:
	pytest unit_tests/

quality_checks:
	isort .
	black .
	pylint --recursive=y .

build: quality_checks test
	docker build -t ${LOCAL_IMAGE_NAME} .

integration_test: build
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash integration_test/run.sh

deploy: test quality_checks build integration_test
	docker run -it --rm -p 9696:9696 \
    -e S3_BUCKET_PATH=$S3_BUCKET_PATH\
    -e MLFLOW_EXPERIMENT_ID=$MLFLOW_EXPERIMENT_ID \
    -e RUN_ID=$RUN_ID \
    house-price-prediction-service:v3

setup:
	bash .env
	pipenv install
	pipenv shell
	pre-commit install
