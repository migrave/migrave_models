import os
import yaml
import joblib
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Cols to fill with max or min 
NAN_MAX_COLS = ['of_gaze_0_x',
                'of_gaze_0_y',
                'of_gaze_0_z',
                'of_gaze_1_x',
                'of_gaze_1_y',
                'of_gaze_1_z',
                'of_gaze_angle_x',
                'of_gaze_angle_y',
                'of_gaze_distance',
                'of_gaze_distance_x',
                'of_gaze_distance_y',
                'of_pose_Rxv',
                'of_pose_Ry',
                'of_pose_Rz',
                'of_pose_Tx',
                'of_pose_Ty',
                'of_pose_Tz',
                'of_pose_distance']

def parse_yaml_config(config_file):
    if config_file and os.path.isfile(config_file):
        configs = {}
        with open(config_file, 'r') as infile:
            configs = yaml.safe_load(infile)
    
        return configs 
    else:
        print("Config not found or not given")

def save_classifier(classifier, mean, std, classifier_name):
    with open(classifier_name, 'wb') as f:
        joblib.dump([classifier, mean, std], f, protocol=2)

def standardize_data(train_data, test_data):
    """
    standardized data and fill nan with max value in the corr. col
    input: Raw train_data and test_data (in pandas dataframe format)
    output: standarfized train_data and test_data
    """
    train_data_mean = []
    train_data_std = []
    for c in train_data.columns:
        # compute man and std while ignoring nan
        mean = np.nanmean(train_data[c])
        std = np.nanstd(train_data[c])
        
        train_data_mean.append(mean)
        train_data_std.append(std)

        if std == 0:
            train_data[c] = (train_data[c]-mean)
            test_data[c] = (test_data[c]-mean)
        else:
            train_data[c] = (train_data[c]-mean)/(std)
            test_data[c] = (test_data[c]-mean)/(std)
        
        # fill nan with min if column not in NAN_MAX_COLS, otherwise fill with max
        if c not in NAN_MAX_COLS:
            min_val = np.nanmin(train_data[c])
            train_data[c] = train_data[c].fillna(min_val)
            test_data[c] = test_data[c].fillna(min_val)
        else:
            max_val = np.nanmax(train_data[c])
            train_data[c] = train_data[c].fillna(max_val)
            test_data[c] = test_data[c].fillna(max_val)
            
    return train_data, test_data, np.asarray(train_data_mean), np.asarray(train_data_std)

def split_generalized_data(dataframe, idx, non_feature_cols=None):
    """
    train on one of the user, and test for others
    Input:
        dataframe: dataset
        idx: Index of participat to be used as test set
    Return:
        train_data, train_labels, test_data, test_labels, train_data_mean, train_data_std
    """
    data = dataframe.copy()
    data = data.sort_values(['session_num', 'timestamp'], ascending=[True, True])
    
    train_data = data.loc[data["participant"] != idx]
    test_data = data.loc[data["participant"] == idx]
    
    train_labels = train_data['engagement']
    test_labels = test_data['engagement']
    
    if non_feature_cols:
        train_data = train_data.drop(columns=non_feature_cols)
        test_data = test_data.drop(columns=non_feature_cols)
    else:
        train_data = train_data.drop(columns=['engagement', 'session_num', 'participant', 'timestamp'])
        test_data = test_data.drop(columns=['engagement', 'session_num', 'participant', 'timestamp'])
    
    # shuffle data
    train_data, train_labels = shuffle(train_data, train_labels)
    
    train_data, test_data, mean, std = standardize_data(train_data, test_data)
    
    return train_data, train_labels, test_data, test_labels, mean, std

def split_individualized_data(dataframe,
                              idx,
                              train_percentage,
                              non_feature_cols=None):
    
    data = dataframe.loc[dataframe["participant"] == idx].copy()
        
    # before split, sort value based on session_num and timestamp
    data = data.sort_values(['session_num', 'timestamp'], ascending=[True, True])
    labels = data['engagement']
    
    test_split_size = 1.0 - train_percentage
    train_data, test_data, train_labels, test_labels = train_test_split(data, 
                                                                        labels, 
                                                                        test_size=test_split_size, 
                                                                        shuffle=True)
    if non_feature_cols:
        train_data = train_data.drop(columns=non_feature_cols)
        test_data = test_data.drop(columns=non_feature_cols)
    else:
        train_data = train_data.drop(columns=['engagement', 'session_num', 'participant', 'timestamp'])
        test_data = test_data.drop(columns=['engagement', 'session_num', 'participant', 'timestamp'])
    
    train_data, test_data, mean, std = standardize_data(train_data, test_data)
    
    return train_data, train_labels, test_data, test_labels, mean, std
