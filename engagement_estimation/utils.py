from typing import Dict, Tuple
import os

import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras


MIGRAVE_VISUAL_FEATURES = ['of_AU01_c', 'of_AU02_c', 'of_AU04_c', 'of_AU05_c',
                           'of_AU06_c', 'of_AU07_c', 'of_AU09_c', 'of_AU10_c', 'of_AU12_c',
                           'of_AU14_c', 'of_AU15_c', 'of_AU17_c', 'of_AU20_c', 'of_AU23_c',
                           'of_AU25_c', 'of_AU26_c', 'of_AU28_c', 'of_AU45_c', 'of_gaze_0_x',
                           'of_gaze_0_y', 'of_gaze_0_z', 'of_gaze_1_x', 'of_gaze_1_y',
                           'of_gaze_1_z', 'of_gaze_angle_x', 'of_gaze_angle_y', 'of_pose_Tx',
                           'of_pose_Ty', 'of_pose_Tz', 'of_pose_Rx', 'of_pose_Ry', 'of_pose_Rz',
                           "of_confidence", "of_success", "of_ts_success", "of_pose_distance", "of_gaze_distance",
                           "of_gaze_distance_x", "of_gaze_distance_y", "of_confidence_var", "of_ts_success_var",
                           "of_pose_distance_var", "of_gaze_distance_var", "of_gaze_distance_x_var",
                           "of_gaze_distance_y_var", "of_pose_Rx_var", "of_pose_Ry_var", "of_pose_Rz_var",
                           "of_pose_Tx_var", "of_pose_Ty_var", "of_pose_Tz_var", "of_gaze_0_x_var", "of_gaze_0_y_var",
                           "of_gaze_0_z_var", "of_gaze_1_x_var", "of_gaze_1_y_var", "of_gaze_1_z_var",
                           "of_gaze_angle_x_var", "of_gaze_angle_y_var", "of_success_change", "of_AU01_c_change",
                           "of_AU02_c_change", "of_AU04_c_change", "of_AU05_c_change", "of_AU06_c_change",
                           "of_AU07_c_change", "of_AU09_c_change", "of_AU10_c_change", "of_AU12_c_change",
                           "of_AU14_c_change", "of_AU15_c_change", "of_AU17_c_change", "of_AU20_c_change",
                           "of_AU23_c_change", "of_AU25_c_change", "of_AU26_c_change", "of_AU28_c_change",
                           "of_AU45_c_change", "op_person_n_col", "op_person_n_col_change"]
MIGRAVE_AUDIAL_FEATURES = ["a_harmonicity", "a_intensity", "a_mfcc_0", "a_mfcc_1", "a_pitch_frequency",
                           "a_pitch_strength", "a_harmonicity_var", "a_intensity_var", "a_mfcc_0_var",
                           "a_mfcc_1_var", "a_pitch_frequency_var", "a_pitch_strength_var"]
MIGRAVE_GAME_FEATURES = ["ros_aptitude", "ros_diff_1", "ros_diff_2", "ros_games_session", "ros_in_game",
                         "ros_mistakes_game", "ros_mistakes_session", "ros_skill_EM", "ros_skill_IM", "ros_ts_attempt",
                         "ros_ts_game_start", "ros_ts_robot_talked", "ros_aptitude_var"]
NON_FEATURES_COLS = ["participant", "session_num", "timestamp", "engagement", "index_original", "frame_number", "date_time"]

JOIN_FEATURES_COLS = ["participant", "session_num", "index_original", "frame_number"]

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


MIGRAVE_GREY = [200, 200, 200]
MIGRAVE_RED = [234, 74, 82]
MIGRAVE_GREEN = [73, 164, 100]
MIGRAVE_BLUE = [105, 172, 211]
MIGRAVE_ORANGE = [255, 115, 74]
MIGRAVE_PALETTE = [MIGRAVE_RED, MIGRAVE_GREEN, MIGRAVE_BLUE, MIGRAVE_ORANGE]


def save_classifier(classifier, mean, std, classifier_name):
    """
    Save classifier
    Input:
      classifier: classifier to save
      mean: mean of train data
      std: standar dev of train data
      classifier_name: file name of the classifier
    """
    if isinstance(classifier, keras.Sequential):
        classifier.save(classifier_name + ".h5")
        with open(classifier_name + ".joblib", 'wb') as f:
            joblib.dump([mean, std], f, protocol=2)
    else:
        with open(classifier_name + ".joblib", 'wb') as f:
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


def split_generalized_data(dataframe, idx, non_feature_cols=None, sequence_model=False):
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

    train_labels = train_data[['engagement']]
    test_labels = test_data[['engagement']]

    if non_feature_cols:
        train_data = train_data.drop(columns=non_feature_cols)
        test_data = test_data.drop(columns=non_feature_cols)
    else:
        train_data = train_data.drop(columns=NON_FEATURES_COLS)
        test_data = test_data.drop(columns=NON_FEATURES_COLS)

    train_data, train_mean, train_std = standardize_data(train_data)
    test_data, _, _ = standardize_data(test_data, mean=train_mean, std=train_std)

    if sequence_model:
        data = data.sort_values(["participant", 'session_num', 'timestamp'], ascending=[True, True, True])
        session_groups = data.groupby(["participant", 'session_num'])
        session_sequences = [list(group.index.values) for name, group in session_groups]
        train_data = [train_data.loc[session_sequence].values for session_sequence in session_sequences if session_sequence[0] in list(train_data.index.values)]
        test_data = [test_data.loc[session_sequence].values for session_sequence in session_sequences if session_sequence[0] in list(test_data.index.values)]
        train_labels = [train_labels.loc[session_sequence].values for session_sequence in session_sequences if session_sequence[0] in list(train_labels.index.values)]
        test_labels = [test_labels.loc[session_sequence].values for session_sequence in session_sequences if session_sequence[0] in list(test_labels.index.values)]
    else:
        train_data = train_data.values
        test_data = test_data.values
        train_labels = train_labels.values
        test_labels = test_labels.values
    # shuffle data
    train_data, train_labels = shuffle(train_data, train_labels)

    return train_data, train_labels, test_data, test_labels, train_mean, train_std


def split_individualized_data(dataframe,
                              idx,
                              train_percentage,
                              non_feature_cols=None,
                              sequence_model=False):
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
    labels = data[['engagement']]

    test_split_size = 1.0 - train_percentage

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_split_size, shuffle=False)

    if non_feature_cols:
        train_data = train_data.drop(columns=non_feature_cols)
        test_data = test_data.drop(columns=non_feature_cols)
    else:
        train_data = train_data.drop(columns=NON_FEATURES_COLS)
        test_data = test_data.drop(columns=NON_FEATURES_COLS)

    train_data, train_mean, train_std = standardize_data(train_data)
    test_data, _, _ = standardize_data(test_data, mean=train_mean, std=train_std)

    if sequence_model:
        session_groups = data.groupby(["participant", 'session_num'])
        session_sequences = [list(group.index.values) for name, group in session_groups]
        train_data = [train_data.loc[[idx for idx in session_sequence if idx in list(train_data.index.values)]].values for session_sequence in session_sequences if
                      [idx for idx in session_sequence if idx in list(train_data.index.values)]]
        test_data = [test_data.loc[[idx for idx in session_sequence if idx in list(test_data.index.values)]].values for session_sequence in session_sequences if
                     [idx for idx in session_sequence if idx in list(test_data.index.values)]]
        train_labels = [train_labels.loc[[idx for idx in session_sequence if idx in list(train_labels.index.values)]].values for session_sequence in session_sequences if
                        [idx for idx in session_sequence if idx in list(train_labels.index.values)]]
        test_labels = [test_labels.loc[[idx for idx in session_sequence if idx in list(test_labels.index.values)]].values for session_sequence in session_sequences if
                       [idx for idx in session_sequence if idx in list(test_labels.index.values)]]
    else:
        train_data = train_data.values
        test_data = test_data.values
        train_labels = train_labels.values
        test_labels = test_labels.values

    train_data, train_labels = shuffle(train_data, train_labels)

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

    for metric, means in results.items():
        fig, ax = plt.subplots()
        size = 0.3
        cmap = plt.get_cmap("BrBG")

        # reorder results based on auroc
        means = dict(sorted(means.items(), key=lambda item: item[1], reverse=True))
        num_of_plots = len(means)
        migrave_idx = cmap_idx % len(MIGRAVE_PALETTE)
        color_gradient = get_color_gradient([255, 255, 255], MIGRAVE_PALETTE[migrave_idx], resolution=100)
        legend_names = ["_Hidden"]*num_of_plots*2
        for i,clf in enumerate(means.keys()):
            ax.pie([100-means[clf], means[clf]], radius=3-i*size,
                    colors=[cmap(128), color_gradient[int(means[clf])]], startangle=90,
                    wedgeprops=dict(width=size, edgecolor='w'))
            legend_names[i+i+1] = clf + f" ({means[clf]}%)"

        ax.set(aspect="equal")
        plt.rcParams['font.size'] = 14
        plt.title(f"{metric} on {name} models", y=1.35)
        plt.legend(legend_names, loc=(1.5, 0.5), title="Models")
        plt.savefig(os.path.join(imdir, "_".join([name, metric]) +".png"), bbox_inches='tight')
        plt.savefig(os.path.join(imdir, "_".join([name, metric]) +".pdf"), bbox_inches='tight')
        if show:
            plt.show()


def get_color_gradient(color_1, color_2, resolution=100):
    """
    Returns a color gradient with n colors from tow RGB colors.
    """
    color_1_norm = np.array(color_1) / 255
    color_2_norm = np.array(color_2) / 255
    resolution_steps = np.linspace(0, 1, num=resolution, endpoint=True)
    rgb_colors = [((1 - step) * color_1_norm + (step * color_2_norm)) for step in resolution_steps]
    return [matplotlib.colors.to_hex(rgb_color) for rgb_color in rgb_colors]

