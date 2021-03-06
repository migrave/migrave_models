# Engagement estimation

This package implements engagement classifiers as described in

```
S. Jain, B. Thiagarajan, Z. Shi, C. Clabaugh, and M. J. Matarić, "Modeling engagement in
long-term, in-home socially assistive robot interventions for children with autism
spectrum disorders," Science Robotics, vol. 5, no. 39, 2020.
```

Engagement estimation models can be trained in either a *generalised* mode, namely on data from multiple users, or a *personalised* mode, namely on data from a single user.

We model engagement at three different levels, with labels `-1`, `0`, and `1`:
* `low engagament`: -1
* `neutral engagement`: 0
* `high engagement`: 1

The package is, however, general enough to be used with different engagement categorisations.

## Installing requirements

The requirements can be installed by running
```
pip3 install -r requirements.txt
```

## Training an engagement classifier

The `train_classifier.py` script can be used to train an engagement classifier. The classifier is saved as a joblib file that includes the model as well as the mean and standard deviations of the training data.

The general usage and expected arguments of the training script are described below:

```
usage: train_classifer.py --model MODEL_NAME
                          --type MODEL_TYPE
                          --dataset DATASET_PATH
                          --model-path RESULTING_MODEL_PATH
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
python3 train_classifer.py -m xgboost -t generalised \
        -d dataset/migrave_engagement_data_small.csv -p my_model.joblib
```

An example call for training a personalised model is given below:
```
python3 train_classifer.py -m xgboost -t personalised \
        -d dataset/migrave_engagement_data_small.csv -p my_model_1.joblib -u 1
```

## Testing an engagement classifier

To test a trained engagement classifier, the `test_classifier.py` script can be used. The usage of the script is described below:

```
usage: test_classifer.py --model-path TRAINED_MODEL_PATH
                          --dataset EVALUATION_DATASET_PATH

The arguments are described below:
  --model-path -p
              Path to a trained model (in joblib format)
  --dataset -d
              Path to a CSV dataset used for model testing
```

An example call for testing a model is given below:

```
python3 test_classifier.py -p my_model.joblib -d dataset/migrave_engagement_data_small.csv
```

## Training and evaluating different engagement estimation models

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
