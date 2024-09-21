# Midterm Project

![dataset-cover](https://github.com/joweyel/midterm-project-machine-learning-zoomcamp/assets/50589933/9665c838-4483-4f85-a4ea-5909a2e31bb8)
Image source: [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)

This Git-Repository contains my Midterm Project for the online Class [`Machine Learning Zoomcamp`](https://github.com/DataTalksClub/machine-learning-zoomcamp) by DataTalksClub / Alexey Grigorev. I was able to put everything I have learned in previous weeks of the course to work, to create a machine learning project.


## Problem Description

The project presented here is a classification problem in the medical context of `heart disease prediction`. The dataset was taken from Kaggle and contains anonymous data from 70000 patients. The goal of the dataset is, as the name implies, to predict heart diseases based on the features and target variable listed below. A machine-learning model should be able to confidently predict the presence of a cardiovascular diesease in a patient. 

- The dataset can be found here: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset.

The dataset has the following features:

| **`Feature Name`** | **`Type`**           | **`Meaning`**                          |
| ------------------ | -------------------- | -------------------------------------- |
| `age`              | int (days)           | Age of patient in days                 |
| `height`           | int (cm)             | Height of patient                      |
| `weight`           | float (kg)           | Weight of patient                      |
| `gender`           | categorical          | Gender of patient (1: feamale 2:male)  |
| `ap_hi`            | int                  | Systolic blood pressure of patient     |
| `ap_lo`            | int                  | Diastolic blood pressure of patient    |
| `cholesterol`      | categorical          | Colesterol level of patient <br>(1: normal, 2: above normal, 3: well above normal) |
| `gluc`             | categorical          | Blood sugar (glucose) level of patient <br>(1: normal, 2: above normal, 3: well above normal) |
| `smoke`            | categorical          | Does a patient smoke (0: non-smoker, 1: smoker) |
| `alco`             | categorical          | Alcohol intake of patient (0: non-drinker, 1: drinker) |
| `active`           | categorical          | Physical activity of patient (0: in-active, 1: active) |
| `cardio`           | categorical (target) | Presence or absence of cardiovascular disease (0: absent, 1: present) |


## How to tackle the problem
1. The first step is to clean and pre-process the dataset followed by EDA. The steps taken for this can be found in the jupyter notebook accompanying this project [here](./notebook.ipynb).
2. Three Different Models for binary classifation are trained and evaluated (can also be found in the notebook [here](./notebook.ipynb)):
    - `LogisticRegression`
    - `RandomForestClassifier`
    - `xgboost` with `binary:logistic` objective
3. The best model was chosen and the relevant code for training was exported to a dedicated training-script [train.py](./train.py)
4. The trained model can now be deployed e.g. with `Docker` after creating a docker container from [Dockerfile](./Dockerfile)


## Preparing the environment
In order to run the code of this project you have to create a separated python environment. This can be done either with `virtualenv`, `venv` or `pipenv`. The installation of packages is however always done with the requirements-file. `Python 3.10` is used for every created python environment. This environment is used for local code s.t. data-querys can be sent to the model and to initialize the docker container for the model.
```bash
## Option 1: Virtualenv
virtualenv -p python3.10 project
source project/bin/activate

## Option 2: venv
python3.10 -m venv project
source project/bin/activate


## Option 3: pipenv (recommended)
pipenv --python 3.10 shell

# Installing all dependencies
pip install -r requirements.txt 
```

If you want to run the Notebook of this project yourself you need to install jupyter notebook dependencies inside the pipenv
```bash
pipenv shell
pip install jupyter notebook ipython
```

## Deploying the trained model locally with Docker
To make the trained machine learning model ready for deployment, it will be packaged inside a Docker container. In order to build the docker container you have to execute the following line of code:
```bash
docker build -t midterm-container .
```

To start the docker container run the following command:
```bash
docker run -it --rm -p 9696:9696 midterm-container
```

Now you have to open a new terminal and activate the pipenv-environment, then you can query the model inside the local docker container with the script [predict-test.py](./predict-test.py), where you can supply a json file as argument (an example can be found in the directory [example_json](./example_json/))
```bash
# Activate environment
pipenv shell
# Querying the dockerized model
python predict-test.py example_json/<example>.json
```

## Access the model as Web-App over AWS ElasticBeanstalk

### Deploy model to AWS Elastic Beanstalk
- **Requirement**: AWS IAM account with sufficient `S3`, `EC2` access rights like and `AdministratorAccess-AWSElasticBeanstalk`

Installing and using AWS Elastic Beanstalk inside pipenv environment:
```bash
# Installin EB as Development dependency
pipenv install awsebcli --dev
# Entering the virtual environment
pipenv shell
```

Create service and launch it:
```bash
# Initialize the service in the current folder
eb init -p docker -r <region> predict-serving
# Create and launch
eb create predict-serving-env
```
Obtaining the URL of the service:
```bash
export EB_ENDPOINT=$(aws elasticbeanstalk describe-environments --environment-names predict-serving-env --query "Environments[0].CNAME" --output text)
```
This sets the `EB_ENDPOINT` environment variable that is accessed in the following script.

### Get predictions from the deployed model

The next steps here are pretty simple. This version of querying the trained model, only involves calling a specific predict-script [predict-cloud.py](./predict-cloud.py) for the cloud-version of the model. To do this you have to use the following command:

```bash
python predict-cloud.py example_json/<example>.json
```
The URL to access the web-service is automatically set when you used `export EB_ENDPOINT=...`, otherwise the script will immediately terminate.

### Stopping and removing the deployed model
```bash
eb terminate predict-serving-env
```
