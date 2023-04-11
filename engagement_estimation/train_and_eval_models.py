from typing import Sequence, Dict, List

import os
import sys
import shutil
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

ALLOWED_MODALITIES = ["video", "audio", "game"]


def train_generalized_model(df_data: pd.core.frame.DataFrame,
                            classifier_name: str,
                            participants: np.ndarray=[1,2,3,4],
                            minority_weight_factor=1,
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
    sequence_model = classifier_name in models.SEQUENTIAL_CLASSIFIERS
    evaluation_results = []
    for p in participants:
        # shuffle all data and reindex
        df_data = df_data.reindex(np.random.permutation(df_data.index))
        df_data = df_data.reset_index(drop=True)

        train_data, train_labels, test_data, test_labels, max, min = utils.split_generalized_data(df_data,
                                                                                  idx=p, sequence_model=sequence_model)

        train_unique, train_counts = np.unique(np.concatenate(train_labels).flatten(), return_counts=True)
        test_unique, test_counts = np.unique(np.concatenate(test_labels).flatten(), return_counts=True)
        if len(train_unique) == 1:
            print(
                f"Only one class in train data. Excluded VP{p} for generalized model.")
            continue
        if len(test_unique) == 1:
            print(
                f"Only one class in test data. Excluded VP{p} for generalized model.")
            continue

        try:
            classifier = models.get_classifier(classifier_name, train_counts[np.argmin(train_unique)], train_counts[np.argmax(train_unique)], minority_weight_factor)
        except ValueError as exc:
            Logger.error(str(exc))
            Logger.warning(f"Skipping {classifier_name}")
            continue
        Logger.info(f"Training {classifier_name} on generalized data for participant {p} in test set")

        model, result = models.sklearn(train_data, train_labels,
                                       test_data, test_labels,
                                       classifier, sequence_model, minority_weight_factor)
        if result is None:
            print(f"Faulty prediction. Excluded VP{p} for generalized model.")
            continue
        result['Train'] = ", ".join(str(x) for x in participants if x != p)
        result['Test'] = p
        result["Train_0"] = train_counts[np.argmin(train_unique)]
        result["Train_1"] = train_counts[np.argmax(train_unique)]
        result["Test_0"] = test_counts[np.argmin(test_unique)]
        result["Test_1"] = test_counts[np.argmax(test_unique)]
        result["Total_0"] = result["Train_0"] + result["Test_0"]
        result["Total_1"] = result["Train_1"] + result["Test_1"]

        evaluation_results.append(result)

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        utils.save_classifier(classifier, max, min,
                        "{}/generalized_{}_model_tested_on_{}".format(logdir, classifier_name, p))

    return evaluation_results


def train_individualized_model(df_data: pd.core.frame.DataFrame,
                               classifier_name: str,
                               participants: np.ndarray=[1,2,3,4],
                               train_percentage: Sequence[float]=[0.8, 0.9],
                               minority_weight_factor=1,
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
    sequence_model = classifier_name in models.SEQUENTIAL_CLASSIFIERS
    evaluation_results = []
    for p in participants:
        # shuffle all data and reindex
        df_data = df_data.reindex(np.random.permutation(df_data.index))
        df_data = df_data.reset_index(drop=True)
        for tr_percentage in train_percentage:
            # shuffle all data and reindex
            df_data = df_data.reindex(np.random.permutation(df_data.index))
            df_data = df_data.reset_index(drop=True)

            train_data, train_labels, test_data, test_labels, max, min = utils.split_individualized_data(df_data,
                                                                                                          idx=p,
                                                                                                          train_percentage=tr_percentage,
                                                                                                          sequence_model=sequence_model)
            train_unique, train_counts = np.unique(np.concatenate(train_labels).flatten(), return_counts=True)
            test_unique, test_counts = np.unique(np.concatenate(test_labels).flatten(), return_counts=True)
            if len(train_unique) == 1:
                print(f"Only one class in train data. Excluded VP{p} with train percentage {tr_percentage} for individualized model.")
                continue
            if len(test_unique) == 1:
                print(f"Only one class in test data. Excluded VP{p} with train percentage {tr_percentage} for individualized model.")
                continue
            try:
                classifier = models.get_classifier(classifier_name, train_counts[np.argmin(train_unique)], train_counts[np.argmax(train_unique)], minority_weight_factor)
            except ValueError as exc:
                Logger.error(str(exc))
                Logger.warning(f"Skipping {classifier_name}")
                continue
            Logger.info(f"Training {classifier_name} on individualized data for VP {p} with train percentage {tr_percentage}")

            model, result = models.sklearn(train_data, train_labels,
                                           test_data, test_labels,
                                           classifier, sequence_model, minority_weight_factor)
            if result is None:
                print(f"Faulty prediction. Excluded VP{p} with train percentage {tr_percentage} for individualized model.")
                continue
            result['Participant'] = p
            result['Train'] = "{} ({})".format(p, int(tr_percentage*100))
            result['Test'] = "{} ({})".format(p, int((1-tr_percentage)*100))
            result["Train_0"] = train_counts[np.argmin(train_unique)]
            result["Train_1"] = train_counts[np.argmax(train_unique)]
            result["Test_0"] = test_counts[np.argmin(test_unique)]
            result["Test_1"] = test_counts[np.argmax(test_unique)]
            result["Total_0"] = result["Train_0"] + result["Test_0"]
            result["Total_1"] = result["Train_1"] + result["Test_1"]

            evaluation_results.append(result)

            if not os.path.exists(logdir):
                os.makedirs(logdir)

            utils.save_classifier(classifier, max, min,
                                  "{}/individualized_{}_trained_on_{}_train_percentage_{}".format(logdir,
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

    minority_weight_factor = config["minority_weight_factor"]
    experiment_name = config["experiment_name"]
    datasets = config["datasets"]
    modalities = config["modalities"]
    classifiers = config["models"]
    model_types = config["model_types"]
    dataset_files = [os.path.join("dataset", dataset) for dataset in datasets]
    dataset_stems = []
    features = utils.NON_FEATURES_COLS.copy()

    df_data = pd.read_csv(dataset_files[0], index_col=0).drop(columns=utils.MIGRAVE_VISUAL_FEATURES)
    if "video" in modalities:
        for dataset_file in dataset_files:
            df_data_visual = pd.read_csv(dataset_file, index_col=0)[utils.MIGRAVE_VISUAL_FEATURES + utils.JOIN_FEATURES_COLS]
            dataset_stem = os.path.splitext(os.path.basename(dataset_file))[0]
            dataset_stems.append(dataset_stem)
            df_data_visual = df_data_visual.rename(columns={c: "_".join([c, dataset_stem]) for c in df_data_visual.columns if c in utils.MIGRAVE_VISUAL_FEATURES})
            df_data = df_data.merge(right=df_data_visual, on=utils.JOIN_FEATURES_COLS)
            features.extend(["_".join([c, dataset_stem]) for c in utils.MIGRAVE_VISUAL_FEATURES])
    if "audio" in modalities:
        features.extend(utils.MIGRAVE_AUDIAL_FEATURES)
    if "game" in modalities:
        features.extend(utils.MIGRAVE_GAME_FEATURES)

    dataset_logdir = os.path.join(logdir, experiment_name, "_".join([modality for modality in ALLOWED_MODALITIES if modality in modalities]), "_".join(dataset_stems))
    answer = ""
    print(f"Results will be saved to path {dataset_logdir}")
    if os.path.exists(dataset_logdir):
        print(f"WARNING: Path {dataset_logdir} already exists. Existing files might be overwritten.")
    while answer not in ["Y", "n"]:
        answer = input("Continue? [Y/n]? ")
    if answer == "n":
        sys.exit(0)

    if not os.path.exists(dataset_logdir):
        os.makedirs(dataset_logdir)

    shutil.copyfile(config_path, os.path.join(dataset_logdir, os.path.basename(config_path)))

    participants = np.sort(df_data.participant.unique())

    df_data_copy = df_data[features].copy()

    mean_results = {}
    clf_results = None
    for i, model_type in enumerate(model_types):
        mean_results[model_type] = {"AUROC_1": {}, "AUPRC_1": {}, "AUROC_0": {}, "AUPRC_0": {}}
        for clf_name in classifiers:
            if "generalized" in model_type:
                if len(participants) > 1:
                    clf_results = train_generalized_model(df_data_copy.copy(),
                                                          clf_name,
                                                          minority_weight_factor=minority_weight_factor,
                                                          participants=participants,
                                                          logdir=dataset_logdir)
                else:
                    Logger.warning(f"Number of participant < 2. Skipping training generalized model")
            elif "individualized" in model_type:
                clf_results = train_individualized_model(df_data_copy.copy(),
                                                         clf_name,
                                                         minority_weight_factor=minority_weight_factor,
                                                         participants=participants,
                                                         logdir=dataset_logdir)
            # save results
            if clf_results:
                clf_result_pd = pd.DataFrame(columns=list(clf_results[0].keys()))
                clf_result_pd = clf_result_pd.append(clf_results, ignore_index=True, sort=False).round(3)
                clf_result_pd.to_csv("{}/{}_{}.csv".format(dataset_logdir, model_type, clf_name), index=False)
                mean_results[model_type]["AUROC_1"][clf_name] = round(clf_result_pd.AUROC_1.mean()*100,2)
                mean_results[model_type]["AUPRC_1"][clf_name] = round(clf_result_pd.AUPRC_1.mean() * 100, 2)
                mean_results[model_type]["AUROC_0"][clf_name] = round(clf_result_pd.AUROC_0.mean() * 100, 2)
                mean_results[model_type]["AUPRC_0"][clf_name] = round(clf_result_pd.AUPRC_0.mean() * 100, 2)
            clf_results = None
        # plot results
        if mean_results[model_type]:
            utils.plot_results(mean_results[model_type], cmap_idx=i, name=model_type, imdir=os.path.join(dataset_logdir, "images"), show=False)


if __name__ == '__main__':
    train_and_evaluate(parser.parse_args().config)
