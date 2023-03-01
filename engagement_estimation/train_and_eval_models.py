from typing import Sequence, Dict, List

import os
import warnings
import argparse
import pandas as pd
import numpy as np

import models
import utils
from logger import Logger

# from mas_tools.file_utils import load_yaml_file
import yaml

__author__ = "Mohammad Wasil"

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/config.yaml', help='Config file')


def train_generalized_model(df_data: pd.core.frame.DataFrame,
                            classifier,
                            classifier_name: str,
                            participants: np.ndarray=[1,2,3,4],
                            logdir: str="./logs") -> List[Dict]:
    """Trains a generalized engagement model (i.e. a model trained on date from
    multiple users). Evaluates the trained model using leave-one-out cross validation
    (i.e. the model is evaluated on each participant separately).

    Returns a list of dictionaries containing evaluation results (AUROC, precision, recall,
    F-score, classification accuracy, training participants, and test participant), one
    dictionary per participant; in other words, there are len(participants) dictionaries
    in the resulting list.

    Keyword arguments:
    @param df_data: pd.core.frame.Dataframe -- dataset used for training and testing
    @param classifier -- classifier used for training
    @param classifier_name: str -- name of the classifier
    @param participants: np.ndarray -- participant labels
    @param logdir: str -- directory where the trained models and evaluation results are saved

    """
    evaluation_results = []
    for p in participants:
        # shuffle all data and reindex
        df_data = df_data.reindex(np.random.permutation(df_data.index))
        df_data = df_data.reset_index(drop=True)

        train_data, train_labels, test_data, test_labels, mean, std = utils.split_generalized_data(df_data,
                                                                                  idx=p)
        model, result = models.sklearn(train_data.values, train_labels.values,
                                       test_data.values, test_labels.values,
                                       classifier)
        result['Train'] = ", ".join(str(x) for x in participants if x != p)
        result['Test'] = p

        evaluation_results.append(result)

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        utils.save_classifier(classifier, mean, std,
                        "{}/generalized_{}_model_tested_on_{}.joblib".format(logdir, classifier_name, p))

    return evaluation_results

def train_individualized_model(df_data: pd.core.frame.DataFrame,
                               classifier,
                               classifier_name: str,
                               participants: np.ndarray=[1,2,3,4],
                               train_percentage: Sequence[float]=[0.8, 0.9],
                               logdir: str="./logs") -> List[Dict]:
    """Trains individualized models (one model for each participant),
    using different train/test split percentages.

    Returns a list of dictionaries containing evaluation results (AUROC, precision, recall,
    F-score, classification accuracy, training participants, and test participant), one
    dictionary per participant and train/test percentage split; in other words, there are
        len(participants) * len(train_percentage)
    dictionaries in the resulting list.

    Keyword arguments:
    @param df_data: pd.core.frame.Dataframe -- dataset used for training and testing
    @param classifier -- classifier used for training
    @param classifier_name: str -- name of the classifier
    @param participants: np.ndarray -- participant labels
    @param train_percentage: Sequence[float] -- a list of percentages used for training the model
    @param logdir: str -- directory where the trained models and evaluation results are saved

    """
    evaluation_results = []
    for p in participants:
        # shuffle all data and reindex
        df_data = df_data.reindex(np.random.permutation(df_data.index))
        df_data = df_data.reset_index(drop=True)
        for tr_percentage in train_percentage:
            # shuffle all data and reindex
            df_data = df_data.reindex(np.random.permutation(df_data.index))
            df_data = df_data.reset_index(drop=True)

            train_data, train_labels, test_data, test_labels, mean, std = utils.split_individualized_data(df_data,
                                                                                                          idx=p,
                                                                                                          train_percentage=tr_percentage)
            if len(np.unique(train_labels)) == 1:
                print(f"Only one class in train data. Excluded VP{p} with train percentage {tr_percentage} for individualized model.")
                continue
            if len(np.unique(test_labels)) == 1:
                print(f"Only one class in test data. Excluded VP{p} with train percentage {tr_percentage} for individualized model.")
                continue

            model, result = models.sklearn(train_data.values, train_labels.values,
                                           test_data.values, test_labels.values,
                                           classifier)
            result['Participant'] = p
            result['Train'] = "{} ({})".format(p, int(tr_percentage*100))
            result['Test'] = "{} ({})".format(p, int((1-tr_percentage)*100))

            evaluation_results.append(result)

            if not os.path.exists(logdir):
                os.makedirs(logdir)

            utils.save_classifier(classifier, mean, std,
                                  "{}/individualized_{}_trained_on_{}_train_percentage_{}.joblib".format(logdir,
                                                                                                         classifier_name,
                                                                                                         p,
                                                                                                         tr_percentage))

    return evaluation_results

def train_and_evaluate(config_path: str, logdir: str="./logs") -> None:
    """Trains and evaluates models as specified in the config file under "config_path".
    Saves the results of the training and evaluation in the directory specified by "logdir".

    Keyword arguments:
    @param config_path: str -- path to a config file specifying training parameters,
                               such as classifiers to train, model types (individualised
                               and/or personalised, as well as the dataset to use
                               for training and evaluation)
    @param logdir: str -- directory where training logs and evaluation results will be stored
                          (default './logs')
    """

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    try:
        with open(config_path) as ymlfile:
            config = yaml.safe_load(ymlfile)
    except (OSError, ValueError) as exc:
        Logger.error(str(exc))
        return

    datasets = config["datasets"]
    classifiers = config["models"]
    model_types = config["model_types"]
    dataset_files = [os.path.join("dataset", dataset) for dataset in datasets]
    dataset_stems = []

    df_data = pd.read_csv(dataset_files[0], index_col=0).drop(columns=utils.MIGRAVE_VISUAL_FEATURES)
    for dataset_file in dataset_files:
        df_data_visual = pd.read_csv(dataset_file, index_col=0)[utils.MIGRAVE_VISUAL_FEATURES + utils.JOIN_FEATURES_COLS]
        dataset_stem = os.path.splitext(os.path.basename(dataset_file))[0]
        dataset_stems.append(dataset_stem)
        df_data_visual = df_data_visual.rename(columns={c: "_".join([c, dataset_stem]) for c in df_data_visual.columns if c in utils.MIGRAVE_VISUAL_FEATURES})
        df_data = df_data.merge(right=df_data_visual, on=utils.JOIN_FEATURES_COLS)

    dataset_logdir = os.path.join(logdir, "_".join(dataset_stems))
    if not os.path.exists(dataset_logdir):
        os.makedirs(dataset_logdir)

    participants = np.sort(df_data.participant.unique())

    migrave_visual_features_extended = ["_".join([c, dataset_stem]) for dataset_stem in dataset_stems for c in utils.MIGRAVE_VISUAL_FEATURES]
    features = utils.NON_FEATURES_COLS + migrave_visual_features_extended + utils.MIGRAVE_AUDIAL_FEATURES\
               + utils.MIGRAVE_GAME_FEATURES
    df_data_copy = df_data[features].copy()

    mean_results = {}
    clf_results = None
    for i, model_type in enumerate(model_types):
        mean_results[model_type] = {}
        for clf_name in classifiers:
            try:
                clf = models.get_classifier(clf_name)
            except ValueError as exc:
                Logger.error(str(exc))
                Logger.warning(f"Skipping {clf_name}")
                continue

            Logger.info(f"Training {clf_name} on {model_type} data")
            if "generalized" in model_type:
                if len(participants) > 1:
                    clf_results = train_generalized_model(df_data_copy.copy(),
                                                          clf,
                                                          clf_name,
                                                          participants=participants,
                                                          logdir=dataset_logdir)
                else:
                    Logger.warning(f"Number of participant < 2. Skipping training generalized model")
            elif "individualized" in model_type:
                clf_results = train_individualized_model(df_data_copy.copy(),
                                                         clf,
                                                         clf_name,
                                                         participants=participants,
                                                         logdir=dataset_logdir)
            # save results
            if clf_results:
                clf_result_pd = pd.DataFrame(columns=list(clf_results[0].keys()))
                clf_result_pd = clf_result_pd.append(clf_results, ignore_index=True, sort=False).round(3)
                clf_result_pd.to_csv("{}/{}_{}.csv".format(dataset_logdir, model_type, clf_name), index=False)
                mean_results[model_type][clf_name] = round(clf_result_pd.AUROC.mean()*100,2)
            clf_results = None
        # plot results
        if mean_results[model_type]:
            utils.plot_results(mean_results[model_type], cmap_idx=i, name=model_type, imdir=os.path.join(dataset_logdir, "images"), show=False)


if __name__ == '__main__':
    train_and_evaluate(parser.parse_args().config)
