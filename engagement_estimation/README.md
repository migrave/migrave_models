# Engagement estimation

The engagement is divided into three different levels with labels `-1`, `0`, and `1`:
* `low engagament`: -1
* `neutral engagement`: 0
* `high engagement`: 1

## Install requirements

```
pip3 install -r requirements.txt
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
