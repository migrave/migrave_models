from cleanlab.filter import find_label_issues, get_label_quality_scores
from generate_classifier_outputs import create_cv_predictions, create_cv_voting_predictions
from pathlib import Path
from typing import Union, List, Optional
import sys
import argparse
from functools import reduce
import pandas as pd


def create_label_issues(experiment_dir: Union[str, Path], datasets: List[str], modalities: List[str], classifier_name: str, voting: bool):
    if isinstance(experiment_dir, str):
        experiment_dir = Path(experiment_dir)
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
    classification_dfs.to_csv(experiment_dir / modalities_id / f"{dataset_id}_{classifier_name}_cross_validation_issues.csv")
    print(f"{label_issues.sum() / len(label_issues) * 100:.2f}% of all labels have issues.")


def merge_label_issues(label_issue_files: List[Union[str, Path]], output_dir: Union[str, Path]):
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    label_issue_dfs = []
    file_stems = []
    for label_issue_file in label_issue_files:
        if isinstance(label_issue_file, str):
            label_issue_file = Path(label_issue_file)
        label_issue_df = pd.read_csv(label_issue_file, index_col=0)
        label_issue_dfs.append(label_issue_df)
        file_stems.append(label_issue_file.stem)
    col_names = label_issue_dfs[0].columns
    classification_vote_df = reduce(lambda left, right: pd.merge(left, right, on=["participant", "session_num", "timestamp"], how="left"), label_issue_dfs)
    label_issue_cols = classification_vote_df.filter(regex="^label_issues.*")
    label_issue_merged = label_issue_cols.all(axis=1)
    classification_vote_df["label_issues"] = label_issue_merged
    classification_vote_df = classification_vote_df[col_names]
    classification_vote_df.loc[:, ~classification_vote_df.columns.isin(["participant", "session_num", "timestamp", "label_issues"])] = np.nan
    merge_label_issues_file = output_dir.joinpath("merged_" + "_".join(file_stems) + ".csv")
    classification_vote_df.to_csv(merge_label_issues_file)


class Args:
    experiment_dir = "/home/rfh/Repos/migrave_models/engagement_estimation/logs/exclude_op_of_sucess_ros_scalable/"
    modalities = ["video", "game"]
    datasets = ["features_video_right.csv", "features_video_left.csv", "features_video_color.csv"]
    classifier_name = "-xgboost"
    voting = 1
    command = "create"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="command")
    create_parser = subparser.add_parser("create")
    merge_parser = subparser.add_parser("merge")

    create_parser.add_argument("-ed", "--experiment_dir", type=str, help="Path to the experiment directory")
    create_parser.add_argument("-m", "--modalities", type=str, nargs="+", help="List of modalities")
    create_parser.add_argument("-d", "--datasets", type=str, nargs="+", help="List of datasets")
    create_parser.add_argument("-cn", "--classifier_name", type=str, help="Classifier name")
    create_parser.add_argument("-v", "--voting", type=int, help="Use datasets for separate voting classifiers")

    merge_parser.add_subparsers("-li", "--label_issues", type=str, nargs="+", help="List of label issue files")
    merge_parser.add_subparsers("-o", "--out_dir", type=str, nargs="+", help="Output directory")

    args = parser.parse_args()
    # args = Args
    if args.command == "create":
        for parsed_dir in [args.experiment_dir]:
            if not Path(parsed_dir).is_dir():
                print(f"Parsed directory {parsed_dir} does not exist.")
                sys.exit(0)
        create_label_issues(experiment_dir=args.experiment_dir, datasets=args.datasets, modalities=args.modalities, classifier_name=args.classifier_name, voting=args.voting)
    elif args.command == "merge":
        if not Path(args.out_dir).is_dir():
            print(f"Parsed directory {args.out_dir} does not exist.")
            sys.exit(0)
        for label_issue in [args.label_issues]:
            if not Path(label_issue).is_file():
                print(f"Parsed file {label_issue} does not exist.")
                sys.exit(0)
    else:
        print(f"Expected command to be create or merge but got {args.command} instead")
        sys.exit(0)
