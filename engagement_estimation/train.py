import os
import sys
import warnings
import argparse
import pandas as pd
from sklearn.utils import shuffle

import models
import utils
from logger import Logger

ALLOWED_MODEL_TYPES = ['generalised', 'personalised']

warnings.filterwarnings('ignore')

def train_classifier(model_name: str, model_type: str,
                     dataset_path: str, user_id: int):
    """Trains an engagement classifier.

    Keyword arguments:
    @param model_name: str -- name of the classified to be trained
    @param model_type: str -- model type (generalised or personalised)
    @param dataset_path: str -- path to a CSV dataset used for training
    @param user_id: int -- used for a personalised model; specifies the ID
                           of the user for whom a model should be trained

    """
    data = pd.read_csv(dataset_path)

    model = None
    try:
        model = models.get_classifier(model_name)
    except ValueError as exc:
        Logger.error(str(exc))
        return

    if model_type == 'personalised':
        data = data.loc[data['participant'] == user_id]
    labels = data['engagement']
    data = data.drop(columns=utils.NON_FEATURES_COLS)
    data, data_mean, data_std = utils.standardize_data(data)
    data, labels = shuffle(data, labels)
    model.fit(data, labels)
    return model, data_mean, data_std

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, type=str,
                        help='Name of the model to be trained')
    parser.add_argument('-t', '--type', required=True, type=str,
                        help='Allowed values: {0}'.format(', '.join(ALLOWED_MODEL_TYPES)))
    parser.add_argument('-d', '--dataset', required=True, type=str,
                        help='Path to a CSV dataset')
    parser.add_argument('-p', '--model-path', required=True, type=str,
                        help='Path to a file where the trained model should be saved')
    parser.add_argument('-u', '--user', type=int, default=None,
                        help='Required if type=personalised')

    args = parser.parse_args()
    model_name = args.model
    model_type = args.type
    dataset_path = args.dataset
    user_id = args.user
    model_path = args.model_path

    if model_name not in models.ALLOWED_CLASSIFIERS:
        Logger.error('Model {0} not in allowed classifiers ({1}); exiting'.format(model_name,
                                                                                  ', '.join(models.ALLOWED_CLASSIFIERS)))
        sys.exit(1)

    if model_type not in ALLOWED_MODEL_TYPES:
        Logger.error('Model {0} not in allowed types ({1}); exiting'.format(model_type,
                                                                            ', '.join(ALLOWED_MODEL_TYPES)))
        sys.exit(1)

    if model_type == 'personalised' and not user_id:
        Logger.error('The argument user is required for training a personalised model; exiting')
        sys.exit(1)

    if not os.path.isfile(dataset_path):
        Logger.error(f'{dataset_path} is not a valid file; exiting')
        sys.exit(1)

    if model_type == 'generalised':
        Logger.info(f'Training {model_name} classifier of type {model_type} on dataset {dataset_path}')
    else:
        Logger.info(f'Training {model_name} classifier of type {model_type} for user {user_id} on dataset {dataset_path}')

    classifier, train_data_mean, train_data_std = train_classifier(model_name=model_name,
                                                                   model_type=model_type,
                                                                   dataset_path=dataset_path,
                                                                   user_id=user_id)

    Logger.info(f'Saving model to {model_path}')
    utils.save_classifier(classifier, train_data_mean, train_data_std, model_path)
