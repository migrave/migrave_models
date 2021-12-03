import os
import models
import utils
import argparse
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn import neighbors
import sklearn.ensemble as ensemble
import sklearn.naive_bayes as naive_bayes
import sklearn.calibration as calibration
import sklearn.svm as svm
import sklearn.linear_model as linear_model
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/config.yaml', help='Config file')

MIGRAVE_VISUAL_FEATURES = ['of_AU01_c', 'of_AU02_c', 'of_AU04_c', 'of_AU05_c',
                           'of_AU06_c', 'of_AU07_c', 'of_AU09_c', 'of_AU10_c', 'of_AU12_c',
                           'of_AU14_c', 'of_AU15_c', 'of_AU17_c', 'of_AU20_c', 'of_AU23_c',
                           'of_AU25_c', 'of_AU26_c', 'of_AU28_c', 'of_AU45_c', 'of_gaze_0_x',
                           'of_gaze_0_y', 'of_gaze_0_z', 'of_gaze_1_x', 'of_gaze_1_y',
                           'of_gaze_1_z', 'of_gaze_angle_x', 'of_gaze_angle_y', 'of_pose_Tx',
                           'of_pose_Ty', 'of_pose_Tz', 'of_pose_Rx', 'of_pose_Ry', 'of_pose_Rz']
NON_FEATURES_COLS = ["participant","session_num","timestamp","engagement"]


def train_generalized_model(df_data, 
                            classifier, 
                            participants=[1,2,3,4],
                            logdir="./logs"):
    """
    Train generalized model: leave one out for test
    """
    all_results = []
    for p in participants:
        # shuffle all data and reindex
        df_data = df_data.reindex(np.random.permutation(df_data.index))
        df_data = df_data.reset_index(drop=True)

        train_data, train_labels, test_data, test_labels, mean, std = utils.split_generalized_data(df_data, 
                                                                                  idx=p)
        
        model, result = models.sklearn(train_data, train_labels, test_data, test_labels, classifier)
        result['Train'] = ", ".join(str(x) for x in participants if x != p)
        result['Test'] = p

        all_results.append(result)

        if not os.path.exists(logdir):
            os.makedirs(logdir)
            
        utils.save_classifier(classifier, mean, std, 
                        "{}/generalized_model_tested_on_{}.joblib".format(logdir, p))

    return all_results
  
def train_individualized_model(df_data, 
                               classifier,
                               participants=[1,2,3,4],
                               train_percentage=[0.8, 0.9],
                               logdir="./logs"):
    """
    Train individualized model: train one model for each participant
    starting from 10% of train data, and increment by 10% until 90% according to the paper
    """
    all_results = []
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
            
            model, result = models.sklearn(train_data, train_labels, test_data, test_labels, classifier)
            result['Participant'] = p

            all_results.append(result)
            
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            
            utils.save_classifier(classifier, mean, std, 
                        "{}/individualized_trained_on_{}_train_percentage_{}.joblib".format(logdir, p, tr_percentage))

    return all_results
  
def train(config_path, logdir="./logs"):
    
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    
    if os.path.exists(config_path):
        config = utils.parse_yaml_config(config_path)
    else:
        return
  
    classifiers = config["engagement"]["models"]
    model_types = config["engagement"]["model_types"]
    dataset_file = os.path.join("dataset", config["engagement"]["dataset"])
    df_data = pd.read_csv(dataset_file)

    features = NON_FEATURES_COLS + MIGRAVE_VISUAL_FEATURES
    df_data_copy = df_data[features].copy()

    mean_results = {}
    for model_type in model_types:
        mean_results[model_type] = []
        for clf_name in classifiers:
            if "random_forest" in clf_name:
                clf = ensemble.RandomForestClassifier(n_estimators=100,
                                                      max_depth=None,
                                                      max_features=None, 
                                                      n_jobs=-1)
            elif "xgboost" in clf_name:
                clf = XGBClassifier(n_estimators=100, 
                                    max_depth=6, 
                                    booster='gbtree',
                                    n_jobs=-1, 
                                    eval_metric='logloss')
            elif "adaboost" in clf_name:
                clf = ensemble.AdaBoostClassifier(ensemble.RandomForestClassifier(n_estimators=100))
            elif "svm" in clf_name:
                clf = calibration.CalibratedClassifierCV(svm.LinearSVC()) 
            elif "knn" in clf_name:
                clf = neighbors.KNeighborsClassifier(n_neighbors=5)
            elif "naive_bayes" in clf_name:
                clf = naive_bayes.GaussianNB()
            elif "logistic_regression" in clf_name:
                clf = linear_model.LogisticRegression(penalty='l2', solver='liblinear')

            print(f"Training {clf_name} on {model_type} data")
            if "generalized" in model_type:
                clf_results = train_generalized_model(df_data_copy, clf, logdir=logdir)
            elif "individualized" in model_type:
                clf_results = train_individualized_model(df_data_copy, clf, logdir=logdir)

            # save results
            clf_result_pd = pd.DataFrame(columns=['Train', 'Test', 'Accuracy', 
                                                  'AUROC', 'Precision_0', 'Precision_1', 
                                                  'Recall_0', 'Recall_1', 'F1_0', 'F1_1'])
            clf_result_pd = clf_result_pd.append(clf_results, ignore_index=True, sort=False).round(3)
            clf_result_pd.to_csv("{}/{}_{}.csv".format(logdir, model_type, clf_name), index=False)

            mean_results[model_type].append(round(clf_result_pd.AUROC.mean(),2))
            

if __name__ == '__main__':
    train(parser.parse_args().config)