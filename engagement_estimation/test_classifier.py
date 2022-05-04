import os
import sys
import warnings
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn import metrics

import utils
from logger import Logger

ALLOWED_MODEL_TYPES = ['generalised', 'personalised']

warnings.filterwarnings('ignore')

def test_classifier(model_path: str, dataset_path: str):
    """Tests an engagement classifier on a given dataset.

    Keyword arguments:
    @param model_path: str -- path to a trained model in joblib format
    @param dataset_path: str -- path to a CSV dataset used for testing

    """
    target_map = {0:-1, 1:0, 2:1}
    if os.path.splitext(model_path)[-1].lower() != '.joblib':
        Logger.error(f'The model should be provided in a .joblib format; exiting')
        return

    if os.path.splitext(dataset_path)[-1].lower() != '.csv':
        Logger.error(f'The model should be provided in a .csv format; exiting')
        return

    model = None
    mean = None
    std = None
    with open(model_path, 'rb') as f:
        model, mean, std = joblib.load(f)

    test_data = pd.read_csv(dataset_path)
    test_labels = test_data['engagement']
    test_data = test_data.drop(columns=utils.NON_FEATURES_COLS)
    test_data, _, _ = utils.standardize_data(test_data, mean=mean, std=std)
    test_data, test_labels = shuffle(test_data, test_labels)

    predicted_scores = model.predict_proba(test_data)
    predictions = [target_map[np.argmax(score)] for score in predicted_scores]
    classification_report = metrics.classification_report(test_labels,
                                                          predictions,
                                                          target_names=list(target_map.values()),
                                                          output_dict=True)
    auroc = metrics.roc_auc_score(test_labels, predicted_scores, multi_class="ovr")
    eval_result = {"AUROC": auroc}
    for cls in classification_report.keys():
        if cls in target_map.values():
            eval_result[f"Precision_{cls}"] = classification_report[cls]["precision"]
            eval_result[f"Recall_{cls}"] = classification_report[cls]["recall"]
            eval_result[f"F1_{cls}"] = classification_report[cls]["f1-score"]
        elif cls == "accuracy":
            eval_result["Accuracy"] = classification_report[cls]
    return eval_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--model-path', required=True, type=str,
                        help='Path to a trained model file (in joblib format)')
    parser.add_argument('-d', '--dataset', required=True, type=str,
                        help='Path to a CSV dataset on which the model should be tested')

    args = parser.parse_args()
    model_path = args.model_path
    dataset_path = args.dataset

    if not os.path.isfile(model_path):
        Logger.error(f'{model_path} is not a valid file; exiting')
        sys.exit(1)

    if not os.path.isfile(dataset_path):
        Logger.error(f'{dataset_path} is not a valid file; exiting')
        sys.exit(1)

    Logger.info(f'Evaluating model {model_path} on dataset {dataset_path}')
    evaluation_results = test_classifier(model_path=model_path, dataset_path=dataset_path)
    Logger.info(str(evaluation_results))
