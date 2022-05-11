from typing import Dict, Tuple
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

MIGRAVE_VISUAL_FEATURES = ['of_AU01_c', 'of_AU02_c', 'of_AU04_c', 'of_AU05_c',
                           'of_AU06_c', 'of_AU07_c', 'of_AU09_c', 'of_AU10_c', 'of_AU12_c',
                           'of_AU14_c', 'of_AU15_c', 'of_AU17_c', 'of_AU20_c', 'of_AU23_c',
                           'of_AU25_c', 'of_AU26_c', 'of_AU28_c', 'of_AU45_c', 'of_gaze_0_x',
                           'of_gaze_0_y', 'of_gaze_0_z', 'of_gaze_1_x', 'of_gaze_1_y',
                           'of_gaze_1_z', 'of_gaze_angle_x', 'of_gaze_angle_y', 'of_pose_Tx',
                           'of_pose_Ty', 'of_pose_Tz', 'of_pose_Rx', 'of_pose_Ry', 'of_pose_Rz']
NON_FEATURES_COLS = ["participant", "session_num", "timestamp", "engagement"]

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

def save_classifier(classifier, mean, std, classifier_name):
    """
    Save classifier
    Input:
      classifier: classifier to save
      mean: mean of train data
      std: standar dev of train data
      classifier_name: file name of the classifier
    """
    with open(classifier_name, 'wb') as f:
        joblib.dump([classifier, mean, std], f, protocol=2)

# Some codes are based on
# https://github.com/interaction-lab/exp_engagement/tree/master/Models
def standardize_data(data: pd.core.frame.DataFrame,
                     mean: Dict[str, float] = None,
                     std: Dict[str, float] = None) -> Tuple[pd.core.frame.DataFrame,
                                                            Dict[str, float],
                                                            Dict[str, float]]:
    """Normalises each column with respect to the mean and standard deviation,
    and fills NaN values with the maximum column value. If mean and std are None,
    calculates the column means and standard deviations from the data; otherwise,
    uses the provided values for normalisation.

    Returns:
    * the normalised data
    * a dictionary of column names and mean values (the same as 'mean' if 'mean' is given)
    * a dictionary of column names and standard deviations (the same as 'std' if 'std' is given)

    Keyword arguments:
    @param data: pd.core.frame.DataFrame -- data to be normalised
    @param mean: Dict[str, float] -- dictionary of column names and column means
                                     (default None, in which case the means are
                                      calculated from the data)
    @param std: Dict[str, float] -- dictionary of column names and column standard deviations
                                    (default None, in which case the standard deviations are
                                     calculated from the data)

    """
    data_mean = {}
    data_std = {}
    data_copy = data.copy()
    for c in data.columns:
        # compute man and std while ignoring nan
        if mean is None and std is None:
            col_mean = np.nanmean(data_copy[c])
            col_std = np.nanstd(data_copy[c])
        else:
            col_mean = mean[c]
            col_std = std[c]

        data_mean[c] = col_mean
        data_std[c] = col_std

        if abs(col_std) < 1e-10:
            data_copy[c] = data_copy[c] - col_mean
        else:
            data_copy[c] = (data_copy[c] - col_mean) / col_std

        # fill nan with min if column not in NAN_MAX_COLS, otherwise fill with max
        if c not in NAN_MAX_COLS:
            min_val = np.nanmin(data_copy[c])
            data_copy[c] = data_copy[c].fillna(min_val)
        else:
            max_val = np.nanmax(data_copy[c])
            data_copy[c] = data_copy[c].fillna(max_val)

    return data_copy, data_mean, data_std

def split_generalized_data(dataframe, idx, non_feature_cols=None):
    """
    Train on other users
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
        train_data = train_data.drop(columns=NON_FEATURES_COLS)
        test_data = test_data.drop(columns=NON_FEATURES_COLS)

    # shuffle data
    train_data, train_labels = shuffle(train_data, train_labels)

    train_data, train_mean, train_std = standardize_data(train_data)
    test_data, _, _ = standardize_data(test_data, mean=train_mean, std=train_std)

    return train_data, train_labels, test_data, test_labels, train_mean, train_std

def split_individualized_data(dataframe,
                              idx,
                              train_percentage,
                              non_feature_cols=None):
    """
    Train on a subset of user data
    Input:
        dataframe: dataset
        idx: Index of participat to be trained on
        train_percentage: a list of train percentage
    Return:
        train_data, train_labels, test_data, test_labels, train_data_mean, train_data_std
    """
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
        train_data = train_data.drop(columns=NON_FEATURES_COLS)
        test_data = test_data.drop(columns=NON_FEATURES_COLS)

    train_data, train_mean, train_std = standardize_data(train_data)
    test_data, _, _ = standardize_data(test_data, mean=train_mean, std=train_std)

    return train_data, train_labels, test_data, test_labels, train_mean, train_std

def plot_results(results, cmap_idx=0, name="results", imdir="./logs/images", show=False):
    """
    Input:
      result: a dictionary containing mean AUROC for each model trained on
              individualized and generalized models
      cmap_idx: 0 or 1 (0 represents 0-127 BrBG cmap and 1 represents 128-255 BrBG cmap)
      name: name of the file
      imdir: directory where to store the image
    """
    if not os.path.exists(imdir):
        os.makedirs(imdir)

    fig, ax = plt.subplots()
    size = 0.3
    cmap = plt.get_cmap("BrBG")

    # reorder results based on auroc
    results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    num_of_plots = len(results)
    colors = [np.arange(0,num_of_plots)*16, np.flip(np.arange(11,17)*16)][cmap_idx]
    legend_names = ["_Hidden"]*num_of_plots*2
    for i,clf in enumerate(results.keys()):
        ax.pie([100-results[clf], results[clf]], radius=2-i*size,
                colors=cmap([128,colors[i]]), startangle=90,
                wedgeprops=dict(width=size, edgecolor='w'))
        legend_names[i+i+1] = clf + f" ({results[clf]}%)"

    ax.set(aspect="equal")
    plt.rcParams['font.size'] = 14
    plt.title(f"AUROC on {name} models", y=1.35)
    plt.legend(legend_names, loc=(1.5, 0.5), title="Models")
    plt.savefig(os.path.join(imdir, name+".png"), bbox_inches='tight')
    if show:
        plt.show()
