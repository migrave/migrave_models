# Engagement estimation

The engagement is divided into three different levels with labels `-1`, `0`, and `1`:
* `low engagament`: -1
* `neutral engagement`: 0
* `high engagement`: 1

## Install requirements

```
pip3 install -r requirements.txt
```

## Training an engagement classifier

The `train.py` script can be used to train an engagement classifier. The classifier is saved as a joblib file that includes the model as well as the mean and standard deviations of the training data.

The general usage and expected arguments of the training script are described below:

```
usage: train.py --model MODEL_NAME
                --type MODEL_TYPE
                --dataset DATASET_PATH
                --p RESULTING_MODEL_PATH
                [--user USER_ID]

The arguments are described below:
  --model, -m
              Name of the model to be trained
  --type, -t
              Model type (generalised or personalised)
  --dataset -d
              Path to a CSV dataset used for training the model
  --model-path -p
              Path to a file where the trained model should be saved
  --user -u
              Only required and used if type=personalised (default None)
```

An example call for training a generalised model would be as follows:
```
python3 train.py -m xgboost -t generalised \
        -d dataset/migrave_engagement_data_small.csv -p my_model.joblib
```

An example call for training a personalised model is given below:
```
python3 train.py -m xgboost -t personalised \
        -d dataset/migrave_engagement_data_small.csv -p my_model_1.joblib -u 1
```

## Train and evaluate engagement estimation models

To train and evaluate different types of engagement estimation models, the `train_and_eval_models.py` script can be used:
```
python3 train_and_eval_models.py --config ./config/config.yaml
```

The config file under `config/config.yaml` contains information about what models to train, the model types (`individualized` or `generalized`), and the dataset to use for training and evaluation. An example of a valid config file is given below:

```
model_types:
    - individualized
    - generalized
models:
    - random_forest
    - xgboost
    - svm
    - knn
    - naive_bayes
    - logistic_regression
dataset: migrave_engagement_data_small.csv
```
