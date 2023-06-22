from typing import Dict, Tuple, List
import os

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import matplotlib
import re

matplotlib.use('Agg')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import shuffle
from tensorflow import keras

ALLOWED_CLASSIFIERS = ['random_forest', 'xgboost', 'adaboost', 'svm',
                       'knn', 'naive_bayes', 'logistic_regression', "neural_network",
                       "recurrent_neural_network", "hmm", "crf", "catboost", "lightgbm"]

SEQUENTIAL_CLASSIFIERS = ["recurrent_neural_network", "hmm", "crf"]

KERAS_CLASSIFIERS = ["recurrent_neural_network", "hmm"]

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
NON_FEATURES_COLS = ["participant", "session_num", "timestamp", "engagement", "index_original", "frame_number",
                     "date_time"]

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

ALLOWED_DATASETS = ["features_video_left", "features_video_right", "features_video_color"]

MIGRAVE_GREY = [200, 200, 200]
MIGRAVE_RED = [234, 74, 82]
MIGRAVE_GREEN = [73, 164, 100]
MIGRAVE_BLUE = [105, 172, 211]
MIGRAVE_ORANGE = [255, 115, 74]
MIGRAVE_PALETTE = [MIGRAVE_RED, MIGRAVE_BLUE, MIGRAVE_GREEN, MIGRAVE_ORANGE]


def save_classifier(classifier, max, min, classifier_name):
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
            joblib.dump([max, min], f, protocol=2)
    else:
        with open(classifier_name + ".joblib", 'wb') as f:
            joblib.dump([classifier, max, min], f, protocol=2)


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
        if not c.startswith(tuple(NAN_MAX_COLS)):
            min_val = np.nanmin(data_copy[c])
            data_copy[c] = data_copy[c].fillna(min_val)
        else:
            max_val = np.nanmax(data_copy[c])
            data_copy[c] = data_copy[c].fillna(max_val)

    return data_copy, data_mean, data_std


def normalize_data(data: pd.core.frame.DataFrame,
                   max: Dict[str, float] = None,
                   min: Dict[str, float] = None) -> Tuple[pd.core.frame.DataFrame,
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
    data_max = {}
    data_min = {}
    data_copy = data.copy()
    for c in data.columns:
        # compute man and std while ignoring nan
        if max is None and min is None:
            col_max = np.nanmax(data_copy[c])
            col_min = np.nanmin(data_copy[c])
        else:
            col_max = max[c]
            col_min = min[c]
            data_copy[c].clip(col_min, col_max, inplace=True)

        data_max[c] = col_max
        data_min[c] = col_min

        if abs(col_max - col_min) < 1e-10:
            data_copy[c] = (data_copy[c] - col_min)
        else:
            data_copy[c] = (data_copy[c] - col_min) / (col_max - col_min)

        # fill nan with min if column not in NAN_MAX_COLS, otherwise fill with max
        if not c.startswith(tuple(NAN_MAX_COLS)):
            min_val = np.nanmin(data_copy[c])
            data_copy[c] = data_copy[c].fillna(min_val)
        else:
            max_val = np.nanmax(data_copy[c])
            data_copy[c] = data_copy[c].fillna(max_val)

    return data_copy, data_max, data_min


def split_generalized_data(dataframe, idx, non_feature_cols=None, sequence_model=False):
    """
    Train on other users
    Input:
        dataframe: dataset
        idx: Index of participat to be used as test set
    Return:
        train_data, train_labels, test_data, test_labels, train_max, train_min
    """
    data = dataframe.copy()
    data = data.sort_values(["participant", "session_num", "timestamp"], ascending=[True, True, True])

    train_data = data.loc[data["participant"] != idx]
    test_data = data.loc[data["participant"] == idx]

    train_labels = train_data[["engagement"]]
    test_labels = test_data[["engagement"]]

    if non_feature_cols:
        train_data = train_data.drop(columns=non_feature_cols)
        test_data = test_data.drop(columns=non_feature_cols)
    else:
        train_data = train_data.drop(columns=NON_FEATURES_COLS)
        test_data = test_data.drop(columns=NON_FEATURES_COLS)

    train_data, train_max, train_min = normalize_data(train_data)
    test_data, _, _ = normalize_data(test_data, max=train_max, min=train_min)

    if sequence_model:
        train_data = train_data.join([train_labels, data[["participant", "session_num"]]])
        test_data = test_data.join([test_labels, data[["participant", "session_num"]]])
        train_groups = train_data.groupby(["participant", "session_num"])
        test_groups = test_data.groupby(["participant", "session_num"])
        train_labels = [group[["engagement"]].astype("int64").values for name, group in train_groups]
        test_labels = [group[["engagement"]].astype("int64").values for name, group in test_groups]
        train_data = [group.drop(columns=["participant", "session_num", "engagement"]).values for name, group in
                      train_groups]
        test_data = [group.drop(columns=["participant", "session_num", "engagement"]).values for name, group in
                     test_groups]
    else:
        train_data = train_data.values
        test_data = test_data.values
        train_labels = train_labels.values
        test_labels = test_labels.values
    # shuffle data
    train_data, train_labels = shuffle(train_data, train_labels)

    return train_data, train_labels, test_data, test_labels, train_max, train_min


def split_individualized_data(dataframe, idx, train_percentage, non_feature_cols=None, sequence_model=False):
    """
    Train on a subset of user data
    Input:
        dataframe: dataset
        idx: Index of participant to be trained on
        train_percentage: a list of train percentage
    Return:
        train_data, train_labels, test_data, test_labels, train_max, train_min
    """
    data = dataframe.loc[dataframe["participant"] == idx].copy()
    data = data.sort_values(["session_num", "timestamp"], ascending=[True, True])
    labels = data[["engagement"]]

    test_split_size = 1.0 - train_percentage

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_split_size,
                                                                        shuffle=False)

    if non_feature_cols:
        train_data = train_data.drop(columns=non_feature_cols)
        test_data = test_data.drop(columns=non_feature_cols)
    else:
        train_data = train_data.drop(columns=NON_FEATURES_COLS)
        test_data = test_data.drop(columns=NON_FEATURES_COLS)

    train_data, train_max, train_min = normalize_data(train_data)
    test_data, _, _ = normalize_data(test_data, max=train_max, min=train_min)

    if sequence_model:
        train_data = train_data.join([train_labels, data[["participant", "session_num"]]])
        test_data = test_data.join([test_labels, data[["participant", "session_num"]]])
        train_groups = train_data.groupby(["participant", "session_num"])
        test_groups = test_data.groupby(["participant", "session_num"])
        train_labels = [group[["engagement"]].astype("int64").values for name, group in train_groups]
        test_labels = [group[["engagement"]].astype("int64").values for name, group in test_groups]
        train_data = [group.drop(columns=["participant", "session_num", "engagement"]).values for name, group in
                      train_groups]
        test_data = [group.drop(columns=["participant", "session_num", "engagement"]).values for name, group in
                     test_groups]
    else:
        train_data = train_data.values
        test_data = test_data.values
        train_labels = train_labels.values
        test_labels = test_labels.values

    train_data, train_labels = shuffle(train_data, train_labels)

    return train_data, train_labels, test_data, test_labels, train_max, train_min


def merge_datasets(dataset_files, modalities, exclude_feature_regex=None, exclude_samples_regex=None, label_issue_file=None):
    dataset_stems = []
    features = NON_FEATURES_COLS.copy()
    df_data = pd.read_csv(dataset_files[0], index_col=0).drop(columns=MIGRAVE_VISUAL_FEATURES)
    if "video" in modalities:
        for dataset_file in dataset_files:
            df_data_visual = pd.read_csv(dataset_file, index_col=0)[MIGRAVE_VISUAL_FEATURES + JOIN_FEATURES_COLS]
            dataset_stem = os.path.splitext(os.path.basename(dataset_file))[0]
            dataset_stems.append(dataset_stem)
            df_data_visual = df_data_visual.rename(
                columns={c: "_".join([c, dataset_stem]) for c in df_data_visual.columns if
                         c in MIGRAVE_VISUAL_FEATURES})
            df_data = df_data.merge(right=df_data_visual, on=JOIN_FEATURES_COLS)
            features.extend(["_".join([c, dataset_stem]) for c in MIGRAVE_VISUAL_FEATURES])
    if "audio" in modalities:
        features.extend(MIGRAVE_AUDIAL_FEATURES)
    if "game" in modalities:
        features.extend(MIGRAVE_GAME_FEATURES)
    df_data = df_data[features]
    if exclude_samples_regex is not None:
        for col in df_data:
            if re.compile(exclude_samples_regex).match(col):
                df_data = df_data[df_data[col].astype(bool)]
    if exclude_feature_regex is not None:
        df_data = df_data.drop(df_data.filter(regex=exclude_feature_regex).columns, axis=1)
    if label_issue_file is not None:
        df_data = remove_label_issues(label_issue_file, df_data)
    dataset_stems = [dataset_stem for dataset_stem in ALLOWED_DATASETS if dataset_stem in dataset_stems]

    return df_data, dataset_stems


def remove_label_issues(label_issue_file: str, dataset_df: pd.DataFrame):
    # remove label issues from datasets
    label_issue_file = os.path.join("dataset", label_issue_file)
    label_issue_df = pd.read_csv(label_issue_file, index_col=0)
    label_issue_df = label_issue_df.loc[label_issue_df["label_issues"] == True]
    data_cols = dataset_df.columns
    dataset_df = dataset_df.reset_index(names="__tmp_index__").merge(right=label_issue_df,
                                                                     on=["participant", "session_num", "timestamp"],
                                                                     how="left", indicator=True).set_index(
        "__tmp_index__")
    dataset_df = dataset_df[dataset_df["_merge"] == "left_only"][data_cols]

    return dataset_df


def create_result(test_labels, predictions, target_names, scores_1, scores_0):
    # classification report
    cls_report = metrics.classification_report(test_labels,
                                               predictions,
                                               target_names=list(target_names.values()),
                                               output_dict=True)
    confusion_mtx = metrics.confusion_matrix(test_labels, predictions)
    auroc_1 = metrics.roc_auc_score(test_labels, scores_1)
    auprc_1 = metrics.average_precision_score(test_labels, scores_1)
    auroc_0 = metrics.roc_auc_score(1 - test_labels, scores_0)
    auprc_0 = metrics.average_precision_score(1 - test_labels, scores_0)

    result = {}
    result["AUROC_1"] = auroc_1
    result["AUPRC_1"] = auprc_1
    result["AUROC_0"] = auroc_0
    result["AUPRC_0"] = auprc_0
    for cls in cls_report.keys():
        if cls in target_names.values():
            result[f"Precision_{cls}"] = cls_report[cls]["precision"]
            result[f"Recall_{cls}"] = cls_report[cls]["recall"]
            result[f"F1_{cls}"] = cls_report[cls]["f1-score"]
        elif cls == "accuracy":
            result["Accuracy"] = cls_report[cls]
    result["C_ij(i=label,j=prediction)"] = confusion_mtx
    return result


def extend_generalized_result(result, participants, p, train_0, train_1, test_0, test_1):
    result['Train'] = ", ".join(str(x) for x in participants if x != p)
    result['Test'] = p
    result["Train_0"] = train_0
    result["Train_1"] = train_1
    result["Test_0"] = test_0
    result["Test_1"] = test_1
    result["Total_0"] = result["Train_0"] + result["Test_0"]
    result["Total_1"] = result["Train_1"] + result["Test_1"]
    return result


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
        max_rad = 3
        size = max_rad / len(ALLOWED_CLASSIFIERS)
        cmap = plt.get_cmap("BrBG")

        # reorder results based on auroc
        means = dict(sorted(means.items(), key=lambda item: item[1], reverse=True))
        num_of_plots = len(means)
        migrave_idx = cmap_idx % len(MIGRAVE_PALETTE)
        color_gradient = get_color_gradient([255, 255, 255], MIGRAVE_PALETTE[migrave_idx], resolution=101)
        legend_names = ["_Hidden"] * num_of_plots * 2
        for i, clf in enumerate(means.keys()):
            ax.pie([100 - means[clf], means[clf]], radius=3 - i * size,
                   colors=[cmap(128), color_gradient[int(means[clf])]], startangle=90,
                   wedgeprops=dict(width=size, edgecolor='w'))
            legend_names[i + i + 1] = clf + f" ({means[clf]}%)"

        ax.set(aspect="equal")
        plt.rcParams['font.size'] = 14
        plt.title(f"{metric} on {name} models", y=1.35)
        plt.legend(legend_names, loc=(1.5, 0.5), title="Models")
        plt.savefig(os.path.join(imdir, "_".join([name, metric]) + ".png"), bbox_inches='tight')
        plt.savefig(os.path.join(imdir, "_".join([name, metric]) + ".svg"), bbox_inches='tight')
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


def get_results(logdir, model_type, metrics):
    results = {metric: {} for metric in metrics}
    for file in os.listdir(logdir):
        if file.endswith(".csv"):
            file_name_chunks = os.path.splitext(os.path.basename(file))[0].split("_")
            file_model_type = file_name_chunks[0]
            if file_model_type == model_type:
                file_model = "_".join(file_name_chunks[1:])
                file_df = pd.read_csv(os.path.join(logdir, file))
                for metric in results.keys():
                    results[metric][file_model] = round(file_df[metric].mean() * 100, 2)
    return results


def plot_from_log(logdir, model_type="generalized", metrics=["AUPRC_0", "AUPRC_1"], exclude_classifiers=["catboost"], cmap_idx=0):
    results = get_results(logdir, model_type, metrics)
    for classifier in exclude_classifiers:
        for metric, scores in results.items():
            scores.pop(classifier, None)
    plot_results(results=results, cmap_idx=cmap_idx, name=model_type, imdir=os.path.join(logdir, "assets"))


def plot_balanced_acc_from_log(logdir, model_type="generalized", exclude_classifiers=["catboost"], cmap_idx=0):
    metrics = ["Recall_1", "Recall_0"]
    results = get_results(logdir, model_type, metrics)
    for classifier in exclude_classifiers:
        for metric, scores in results.items():
            scores.pop(classifier, None)
    results = {"Balanced_Accuracy": {classifier: (results["Recall_1"][classifier] + results["Recall_0"][classifier]) / 2 for classifier in results[metrics[0]].keys()}}
    plot_results(results=results, cmap_idx=cmap_idx, name=model_type, imdir=os.path.join(logdir, "assets"))

plot_balanced_acc_from_log(logdir="/home/rfh/Repos/migrave_models/engagement_estimation/logs/exclude_op_of_sucess_ros_scalable_label_issues/video_game/features_video_color", cmap_idx=2)
plot_from_log(logdir="/home/rfh/Repos/migrave_models/engagement_estimation/logs/exclude_op_of_sucess_ros_scalable_label_issues/video_game/features_video_color", cmap_idx=2)

def plot_summary(logdirs: List[str], out_dir, model_type="generalized",
                 metrics=["Recall_0", "AUPRC_0", "AUROC_0", "Precision_0"]):
    logdirs = [Path(logdir) for logdir in logdirs]
    summary_results = {metric: {} for metric in metrics}
    for modality_dir in logdirs:
        modalities = modality_dir.name.split("_")
        if "video" in modalities:
            for perspective_dir in modality_dir.iterdir():
                if perspective_dir.is_dir():
                    modality_key = "-".join([modality_dir.name, perspective_dir.name])
                    modality_result = get_results(perspective_dir, model_type, metrics)
                    for metric in summary_results.keys():
                        model_max_result = max(modality_result[metric], key=modality_result[metric].get)
                        summary_results[metric]["-".join([modality_key, model_max_result])] = modality_result[metric][
                            model_max_result]
        else:
            modality_key = modality_dir.name
            modality_result = get_results(modality_dir, model_type, metrics)
        for metric in summary_results.keys():
            model_max_result = max(modality_result[metric], key=modality_result[metric].get)
            summary_results[metric]["-".join([modality_key, model_max_result])] = modality_result[metric][
                model_max_result]
    plot_results(results=summary_results, cmap_idx=0, name=model_type, imdir=os.path.join(out_dir, "images_summary"))
