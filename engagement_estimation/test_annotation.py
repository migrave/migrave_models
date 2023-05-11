from cleanlab.filter import find_label_issues, get_label_quality_scores
from cleanlab.dataset import health_summary
from test_model import test_model, get_sessions, get_participants
from utils import merge_datasets, ALLOWED_DATASETS
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Union, List, Optional
import sys
import argparse
import matplotlib
from matplotlib import pyplot as plt
# matplotlib.use("TkAgg")


def create_cv_predictions(experiment_dir: Union[str, Path], datasets: List[str], modalities: List[str], classifier_name: str):
    if isinstance(experiment_dir, str):
        experiment_dir = Path(experiment_dir)
    participant_ids = get_participants(dataset=datasets[0])
    classification_dfs = []
    for participant_id in participant_ids:
        sessions = get_sessions(dataset=datasets[0], participant_id=participant_id)
        for session in sessions:
            classification_df, date_time, modalities_id, dataset_id = test_model(experiment_dir=experiment_dir, modalities=modalities,
                                                    classifier_name=classifier_name, participant_id=participant_id,
                                                    datasets=datasets, session=session)
            classification_df["participant"] = participant_id
            classification_df["session_num"] = session
            classification_dfs.append(classification_df)
    classification_dfs = pd.concat(classification_dfs, axis=0)
    return classification_dfs, modalities_id, dataset_id


def create_cv_voting_predictions(experiment_dir: Union[str, Path], dataset_perspectives: List[str], modalities: List[str], classifier_name: str):
    classification_perspective_dfs = []
    dataset_ids = []
    for dataset_perspective in dataset_perspectives:
        classification_dfs, modalities_id, dataset_id = create_cv_predictions(experiment_dir=experiment_dir, datasets=[dataset_perspective], modalities=modalities, classifier_name=classifier_name)
        classification_perspective_dfs.append(classification_dfs)
        dataset_ids.append(dataset_id)
    classification_voting_df = classification_perspective_dfs[0].copy()
    perspectives_score_1 = pd.concat([classification_perspective_df[["scores_1"]] for classification_perspective_df in classification_perspective_dfs], axis=1)
    perspective_weights = pd.concat([classification_perspective_df[["of_success"]].values for classification_perspective_df in classification_perspective_dfs], axis=1)
    voting_of_success = perspective_weights.any(axis=1)
    perspective_weights.loc[~voting_of_success] = 1
    classification_voting_df["scores_1"] = np.average(perspectives_score_1.values, axis=1, weights=perspective_weights.values)
    classification_voting_df["scores_0"] = 1 - classification_voting_df["scores_1"]
    classification_voting_df["predictions"] = (classification_voting_df["scores_1"] >= .5).astype(int)
    classification_voting_df["of_success"] = voting_of_success
    dataset_ids = [dataset_id for dataset_id in ALLOWED_DATASETS if dataset_id in dataset_ids]
    dataset_voting_id = "voting_" + "_".join(dataset_ids)
    return classification_voting_df, modalities_id, dataset_voting_id


def create_label_issues(experiment_dir: Union[str, Path], datasets: List[str], modalities: List[str], classifier_name: str, voting: bool):
    if voting:
        classification_dfs, modalities_id, dataset_id = create_cv_voting_predictions(experiment_dir=experiment_dir,
                                                                              dataset_perspectives=datasets, modalities=modalities,
                                                                              classifier_name=classifier_name)
    else:
        classification_dfs, modalities_id, dataset_id = create_cv_predictions(experiment_dir=experiment_dir, datasets=datasets, modalities=modalities, classifier_name=classifier_name)
    labels = classification_dfs["labels"].to_numpy()
    pred_probs = classification_dfs[["scores_0", "scores_1"]].to_numpy()
    classification_dfs = classification_dfs.sort_values(["participant", "session_num", "timestamp"],
                                                        ascending=[True, True, True])
    label_issues = find_label_issues(labels=labels, pred_probs=pred_probs)
    label_quality = get_label_quality_scores(labels=labels, pred_probs=pred_probs)

    classification_dfs["label_issues"] = label_issues
    classification_dfs["label_quality"] = label_quality
    experiment_name = experiment_dir.name
    classification_dfs.to_csv(f"./dataset/{experiment_name}_{modalities_id}_{dataset_id}_{classifier_name}_cross_validation_issues.csv")
    print(f"{label_issues.sum() / len(label_issues) * 100:.2f}% of all labels have issues.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-ed", "--experiment_dir", type=str,
                        default="/home/rfh/Repos/migrave_models/engagement_estimation/logs/exclude_op_of_sucess_ros_scalable",
                        help="Path to the experiment directory")
    parser.add_argument("-m", "--modalities", required=True, type=str, nargs="+", help="List of modalities")
    parser.add_argument("-d", "--datasets", required=True, type=str, nargs="+", help="List of datasets")
    parser.add_argument("-cn", "--classifier_name", required=True, type=str, help="Classifier name")
    parser.add_argument("-v", "--voting", required=True, type=int, help="Use datasets for separate voting classifiers")
    args = parser.parse_args()
    # args = Args
    for parsed_dir in [args.experiment_dir]:
        if not Path(parsed_dir).is_dir():
            print(f"Parsed directory {parsed_dir} does not exist.")
            sys.exit(0)
    create_label_issues(experiment_dir=args.experiment_dir, datasets=args.datasets, modalities=args.modalities, classifier_name=args.classifier_name, voting=args.voting)




    # # get out-of-sample predictions
    # participant_ids = np.arange(1, 30)
    # datasets = ["features_video_right.csv", "features_video_left.csv", "features_video_color.csv"]
    # classification_dfs = []
    # for participant_id in participant_ids:
    #     sessions = get_sessions(dataset=datasets[0], participant_id=participant_id)
    #     for session in sessions:
    #         classification_df, _, _, _ = test_model(
    #             experiment_dir="/Users/schanowski_rfh/Documents/Projekte/MigrAVE/Experimente/baseline_19_04_2023_fix_annotation_gaps",
    #             modalities=["video", "audio", "game"], classifier_name="xgboost",
    #             participant_id=participant_id,
    #             datasets=datasets, session=session)
    #         classification_df["participant"] = participant_id
    #         classification_df["session_num"] = session
    #         classification_dfs.append(classification_df)
    # classification_dfs = pd.concat(classification_dfs, axis=0)
    # labels = classification_dfs["labels"].to_numpy()
    # pred_probs = classification_dfs[["scores_0", "scores_1"]].to_numpy()
    # classification_dfs.to_csv("/Users/schanowski_rfh/Documents/Projekte/MigrAVE/Experimente/baseline_19_04_2023_fix_annotation_gaps/xgboost_cross_validation_preds.csv")
    #
    #
    # # get out-of-sample label issues
    # classification_dfs = pd.read_csv(
    #     "/Users/schanowski_rfh/Documents/Projekte/MigrAVE/Experimente/baseline_19_04_2023_fix_annotation_gaps/xgboost_cross_validation_preds.csv",
    # index_col=0)
    # classification_dfs = classification_dfs.sort_values(["participant", "session_num", "timestamp"], ascending=[True, True, True])
    # labels = classification_dfs["labels"].to_numpy()
    # pred_probs = classification_dfs[["scores_0", "scores_1"]].to_numpy()
    #
    # label_issues = find_label_issues(labels=labels, pred_probs=pred_probs)
    # label_quality = get_label_quality_scores(labels=labels, pred_probs=pred_probs)
    # health = health_summary(labels=labels, pred_probs=pred_probs, class_names=["disengagement", "engagement"])
    #
    # classification_dfs["label_issues"] = label_issues
    # classification_dfs["label_quality"] = label_quality
    # classification_dfs.to_csv("/Users/schanowski_rfh/Documents/Projekte/MigrAVE/Experimente/baseline_19_04_2023_fix_annotation_gaps/xgboost_cross_validation_issues.csv")


    # # group out-of-sample label issues to consecutive segments and order by length
    # classification_dfs = pd.read_csv(
    #     "/Users/schanowski_rfh/Documents/Projekte/MigrAVE/Experimente/baseline_19_04_2023_fix_annotation_gaps/xgboost_cross_validation_issues.csv",
    # index_col=0)
    # session_num_change = classification_dfs["session_num"].shift() != classification_dfs["session_num"]
    # classification_dfs["session_num_running"] = session_num_change.cumsum()
    # classification_dfs["issue_groups"] = classification_dfs[["label_issues", "participant", "session_num_running"]].diff().cumsum().fillna(0).sum(axis=1)
    # classification_dfs = classification_dfs.loc[classification_dfs["label_issues"] == True]
    # issue_groups = classification_dfs.groupby(["issue_groups"])
    # issue_groups_size = issue_groups.size().sort_values(ascending=False)
    # issue_group_max = classification_dfs.loc[classification_dfs["issue_groups"] == issue_groups_size.index[1]]

    # fig, ax = plt.subplots(1,1)
    # ax.hist(issue_groups_size.values, bins=list(range(1, max(issue_groups_size.values) + 1)))

    # # remove label issues from datasets
    # label_issue_dfs = pd.read_csv(
    #     "/Users/schanowski_rfh/Documents/Projekte/MigrAVE/Experimente/baseline_19_04_2023_fix_annotation_gaps/xgboost_cross_validation_issues.csv",
    # index_col=0)
    # label_issue_dfs = label_issue_dfs.loc[label_issue_dfs["label_issues"] == True]
    # datasets = ["features_video_right.csv", "features_video_left.csv", "features_video_color.csv"]
    # for dataset in datasets:
    #     dataset_file = Path("./dataset").joinpath(dataset)
    #     dataset_df = pd.read_csv(dataset_file, index_col=0)
    #     data_cols = dataset_df.columns
    #     dataset_df = pd.merge(left=dataset_df, right=label_issue_dfs, on=["participant", "session_num", "timestamp"],
    #                             how="left", indicator=True)
    #     dataset_df = dataset_df[dataset_df["_merge"] == "left_only"][data_cols]
    #     # for index, row in label_issue_dfs.iterrows():
    #     #     label_issue_idx = dataset_df.loc[(dataset_df["participant"] == row["participant"]) & (dataset_df["session_num"] == row["session_num"]) & (dataset_df["timestamp"] == row["timestamp"])].index
    #     #     dataset_df.drop(label_issue_idx, inplace=True)
    #     dataset_dir = dataset_file.parent
    #     dataset_clean_file = dataset_dir.joinpath("_".join([dataset_file.stem, "clean"]) + dataset_file.suffix)
    #     dataset_df.to_csv(dataset_clean_file)
