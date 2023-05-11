from pathlib import Path
from typing import Union, List, Optional
import joblib
import os
import subprocess
import numpy as np
import pandas as pd
from tensorflow import keras
from hmmlearn.hmm import GaussianHMM
from sklearn_crfsuite import CRF
import xgboost
import sys
import cv2
from matplotlib import pyplot as plt
import argparse

from train_and_eval_models import ALLOWED_MODALITIES
from models import KERAS_CLASSIFIERS, SEQUENTIAL_CLASSIFIERS
from utils import merge_datasets, normalize_data, NON_FEATURES_COLS, ALLOWED_DATASETS


def create_feature_list(classifier_file):
    classifier, norm_max, norm_min = joblib.load(classifier_file)
    feature_readme_df = pd.DataFrame({"Features": norm_max.keys(), "Type": norm_max.values(),
                                      "Modality": ["Video" if "video" in key else "Game Performance" if "ros" in key else " " for key in norm_max.keys()],
                                      "Description": norm_max.values()})
    feature_readme_df.to_csv("/Users/schanowski_rfh/Downloads/features_readme.csv")


def get_sessions(dataset: str, participant_id: int):
    """
    Finds all session ids for participant in dataset.
    :param dataset:
    :param participant_id:
    :return:
    """
    dataset_file = os.path.join("./dataset", dataset)
    df_data = pd.read_csv(dataset_file, index_col=0)
    df_data = df_data.loc[df_data["participant"] == participant_id]
    return df_data["session_num"].unique()


def get_participants(dataset: str):
    """
    Finds all session ids for participant in dataset.
    :param dataset:
    :return:
    """
    dataset_file = os.path.join("./dataset", dataset)
    df_data = pd.read_csv(dataset_file, index_col=0)
    return df_data["participant"].unique()


def load_generalized_classifier(experiment_dir: Union[str, Path], modalities: List[str], dataset_stems: List[str],
                                classifier_name: str, participant_id: int):
    """
    Loads classifier which was testet on participant with participant_id in the leave-one-out-CV including the corresponding
    normalization data.
    :param experiment_dir:
    :param modalities:
    :param dataset_stems:
    :param participant_id:
    :return:
    """
    if isinstance(experiment_dir, str):
        experiment_dir = Path(experiment_dir)
    modalities_id = "_".join([modality for modality in ALLOWED_MODALITIES if modality in modalities])
    modality_dir = experiment_dir.joinpath(modalities_id)
    if "video" in modalities:
        dataset_id = "_".join(dataset_stems)
        log_dir = modality_dir.joinpath(dataset_id)
    else:
        dataset_id = None
        log_dir = modality_dir
    model_name = f"generalized_{classifier_name}_model_tested_on_{participant_id}"
    model_files = [path for path in log_dir.iterdir() if path.is_file and path.stem == model_name]
    if classifier_name in KERAS_CLASSIFIERS:
        for file in model_files:
            if file.suffix == ".joblib":
                norm_max, norm_min = joblib.load(file)
            elif file.suffix == ".h5":
                classifier = keras.models.load_model(file)
    else:
        classifier, norm_max, norm_min = joblib.load(model_files[0])

    return classifier, norm_max, norm_min, modalities_id, dataset_id


def get_xgboost_cv_feature_importance(experiment_dir: Union[str, Path], modalities: List[str], datasets: List[str],
                                      participant_ids: List[int]):
    """
    Calculates mean of cross validated feature importances of xgboost models.
    :param experiment_dir:
    :param modalities:
    :param datasets:
    :param participant_ids:
    :return:
    """
    dataset_stems = [os.path.splitext(os.path.basename(dataset))[0] for dataset in datasets]
    dataset_stems = [dataset_stem for dataset_stem in ALLOWED_DATASETS if dataset_stem in dataset_stems]
    feature_importance_dfs = []
    for participant_id in participant_ids:
        classifier, norm_max, norm_min, modalities_id, dataset_id = load_generalized_classifier(
            experiment_dir=experiment_dir, modalities=modalities, dataset_stems=dataset_stems,
            classifier_name="xgboost", participant_id=participant_id)
        feature_importance_df = pd.DataFrame({"features": norm_max.keys(), f"importance_{participant_id}": classifier.feature_importances_})
        feature_importance_dfs.append(feature_importance_df.set_index("features"))
    feature_importance_dfs = pd.concat(feature_importance_dfs, axis=1)
    feature_importance_dfs["mean_importance"] = feature_importance_dfs.mean(axis=1)

    return feature_importance_dfs


def load_data_and_classifier(experiment_dir: Union[str, Path], classifier_name: str, modalities: List[str],
                             datasets: List[str], participant_id: int, session: int, sequence_model: bool):
    """
    Loads the features and classifier with respect to the used modalities and perspectives and filters for the
    participant.
    :param experiment_dir:
    :param classifier_name:
    :param modalities:
    :param datasets:
    :param participant_id:
    :param session:
    :param sequence_model:
    :return:
    """
    dataset_files = [os.path.join("./dataset", dataset) for dataset in datasets]
    features, dataset_stems = merge_datasets(dataset_files, modalities)
    features = features.loc[features["participant"] == participant_id]
    features = features.loc[features["session_num"] == session]
    features = features.sort_values(["timestamp"], ascending=[True])
    timestamps = features[["timestamp"]]
    labels = features[["engagement"]]
    date_time = features["date_time"].iloc[0]

    classifier, norm_max, norm_min, modalities_id, dataset_id = load_generalized_classifier(
        experiment_dir=experiment_dir,
        modalities=modalities,
        dataset_stems=dataset_stems,
        classifier_name=classifier_name,
        participant_id=participant_id)

    of_success = features.filter(regex="^ of_success_features. *").values
    features = features[norm_max.keys()]

    features, _, _ = normalize_data(features, max=norm_max, min=norm_min)
    if sequence_model:
        features = [features.values]
        labels = [labels.values]
    else:
        features = features.values
        labels = labels.values
    return features, labels, of_success, timestamps, date_time, classifier, modalities_id, dataset_id


def test_model(experiment_dir: Union[str, Path], modalities: List[str], classifier_name: str,
               participant_id: int, datasets: List[str], session: int, target_names={0: 0, 1: 1}):
    """
    Predicts probabilities for classes with respect to time.
    :param experiment_dir:
    :param modalities:
    :param classifier_name:
    :param participant_id:
    :param persepctives:
    :param session:
    :return:
    """
    sequence_model = True if classifier_name in SEQUENTIAL_CLASSIFIERS else False
    features, labels, of_success, timestamps, date_time, classifier, modalities_id, dataset_id = load_data_and_classifier(
        experiment_dir=experiment_dir,
        classifier_name=classifier_name,
        modalities=modalities,
        datasets=datasets,
        participant_id=participant_id,
        session=session,
        sequence_model=sequence_model)
    if isinstance(classifier, keras.Sequential):
        scores_1 = classifier.predict(features)
        scores_1 = [score[0] for score in scores_1]
        scores_0 = [1 - score_1 for score_1 in scores_1]
        labels = np.concatenate(labels).flatten()
        predictions = [target_names[np.rint(sc)] for sc in scores_1]
    elif isinstance(classifier, GaussianHMM):
        print("TODO: save the label_0 and label_1 to use model for classification")
        sys.exit(0)
    elif isinstance(classifier, CRF):
        test_data = [[dict(zip(map(str, list(range(len(timestep)))), timestep)) for timestep in sequence] for sequence
                     in features]
        scores = classifier.predict_marginals(test_data)
        scores = np.array([[timestamp["0"], timestamp["1"]] for sequence in scores for timestamp in sequence])
        scores_1 = scores[:, 1]
        scores_0 = scores[:, 0]
        labels = np.concatenate(labels).flatten()
        predictions = [target_names[np.argmax(sc)] for sc in scores]
    elif isinstance(classifier, xgboost.XGBClassifier):
        scores = classifier.predict_proba(features)
        scores_1 = scores[:, 1]
        scores_0 = scores[:, 0]
        predictions = [target_names[np.argmax(sc)] for sc in scores]
    else:
        scores = classifier.predict_proba(features)
        scores_1 = scores[:, 1]
        scores_0 = scores[:, 0]
        predictions = [target_names[np.argmax(sc)] for sc in scores]

    classification_df = timestamps.copy()
    classification_df["scores_0"] = scores_0
    classification_df["scores_1"] = scores_1
    classification_df["predictions"] = predictions
    classification_df["labels"] = labels
    classification_df["of_success"] = of_success

    return classification_df, date_time, modalities_id, dataset_id


def get_video(data_dir: Union[str, Path], participant_id: int, perspective: str, date_time: str):
    """
    Gets video from given perspective for participant and session.
    :param data_dir:
    :param participant_id:
    :param perspective:
    :param session:
    :return:
    """
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    participant_id_dirs = [participant_id_dir for participant_id_dir in data_dir.iterdir() if
                           participant_id_dir.is_dir() and participant_id_dir.name.startswith(f"VP{participant_id:02}")]
    session_dir = None
    for participant_id_dir in participant_id_dirs:
        if participant_id_dir.joinpath(date_time).is_dir():
            session_dir = participant_id_dir.joinpath(date_time)
            break
    video = None
    for video_file in session_dir.iterdir():
        if video_file.is_file() and video_file.suffix == ".mp4":
            if perspective in video_file.name:
                video = video_file
                break

    return video


def generate_prediction_video(experiment_dir: Union[str, Path], data_dir: Union[str, Path],
                              output_dir: Union[str, Path], modalities: List[str], classifier_name: str,
                              participant_ids: List[int], datasets: List[str], sessions: Optional[List[int]] = None):
    """
    Generates video with probabilities for classes as running plot.
    :param experiment_dir:
    :param data_dir:
    :param output_dir:
    :param modalities:
    :param classifier_name:
    :param participant_ids:
    :param datasets:
    :param sessions:
    :return:
    """
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if isinstance(experiment_dir, str):
        experiment_dir = Path(experiment_dir)

    for participant_id in participant_ids:
        if sessions is None:
            sessions = get_sessions(dataset=datasets[0], participant_id=participant_id)
        for session in sessions:
            classification_df, date_time, modalities_id, dataset_id = test_model(experiment_dir=experiment_dir,
                                                                                 modalities=modalities,
                                                                                 classifier_name=classifier_name,
                                                                                 participant_id=participant_id,
                                                                                 datasets=datasets,
                                                                                 session=session)
            perspective = Path(datasets[0]).stem.split("_")[-1]
            input_video_file = get_video(data_dir=data_dir, participant_id=participant_id, perspective=perspective,
                                         date_time=date_time)
            if not input_video_file.is_file():
                print(f"Video file for participant {participant_id} and session {session} does not exist.")
                continue

            output_log_dir = output_dir.joinpath("MigrAVE_Model_Test")
            output_experiment_dir = output_log_dir.joinpath(experiment_dir.name)
            output_modalities_dir = output_experiment_dir.joinpath(modalities_id)
            if dataset_id is None:
                output_video_dir = output_modalities_dir
            else:
                output_video_dir = output_modalities_dir.joinpath(dataset_id)
            output_video_dir.mkdir(parents=True, exist_ok=True)
            output_video_name = Path(f"classifier_{classifier_name}_tested_on_{participant_id}_session_{session}.mp4")
            output_video_file = output_video_dir.joinpath(output_video_name)

            cap = cv2.VideoCapture(str(input_video_file))
            frame_counter = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            width = int(cap.get(3))
            height = int(cap.get(4))
            plot_height = 160
            new_height = height + plot_height
            out = cv2.VideoWriter(str(output_video_file), fourcc, fps, (width, new_height), 1)

            buffer_len = 6

            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    seconds = frame_counter / fps
                    lower_bound = seconds - buffer_len / 2
                    upper_bound = seconds + buffer_len / 2
                    if len(classification_df) > 1:
                        while classification_df["timestamp"].iloc[0] < lower_bound:
                            classification_df.drop(index=classification_df.index[0], axis=0, inplace=True)
                    closest = classification_df.iloc[(classification_df["timestamp"] - upper_bound).abs().argsort()[:1]]
                    buffer_df = classification_df.loc[:closest.index[0], ]
                    buffer_df["timestamp"] = buffer_df["timestamp"] - seconds
                    fig = plt.figure(figsize=(5, 1))
                    ax_0 = fig.add_subplot()
                    ax_0.plot(buffer_df["timestamp"], buffer_df["labels"], color="blue", label="label")
                    ax_0.plot(buffer_df["timestamp"], buffer_df["scores_1"], color="green", label="engaged")
                    ax_0.axvline(x=0, ymin=0, ymax=1, color="black")
                    ax_0.set_xlim([- buffer_len / 2, buffer_len / 2])
                    ax_0.set_ylim([-0.05, 1.05])
                    ax_0.set_xticks([])
                    ax_0.set_yticks([0, 0.5, 1])
                    fig.tight_layout()
                    fig.canvas.draw()
                    fig_arr = np.array(fig.canvas.renderer._renderer)
                    fig_arr = cv2.resize(fig_arr, (width, plot_height))
                    fig_arr = cv2.cvtColor(fig_arr, cv2.COLOR_RGBA2BGR)
                    out.write(np.concatenate((frame, fig_arr), axis=0, dtype=np.uint8))
                    frame_counter += 1
                else:
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()

            output_video_audio_file = "_".join([output_video_name.stem, "audio"]) + output_video_name.suffix
            subprocess.call(
                ["ffmpeg", "-i", f"{output_video_file.name}", "-i", f"{input_video_file}", "-map", "0:v", "-map",
                 "1:a", "-c:v",
                 "copy", "-c:a",
                 "copy", "-shortest", f"{output_video_audio_file}"], cwd=output_video_dir)

            output_video_file.unlink(missing_ok=True)


class Args:
    experiment_dir = "/home/rfh/Repos/migrave_models/engagement_estimation/logs/clean_lab_exclude_issues"
    data_dir = "/media/veracrypt1/MigrAVEProcessed/MigrAVEDaten"
    output_dir = "/media/veracrypt1/MigrAVEProcessed"
    modalities = ["video", "audio", "game"]
    datasets = ["features_video_right.csv", "features_video_left.csv", "features_video_color.csv"]
    participant_ids = [1, 11, 19]
    sessions = None
    classifier_name = "xgboost"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-ed", "--experiment_dir", type=str,
                        default="/home/rfh/Repos/migrave_models/engagement_estimation/logs/clean_lab_exclude_issues",
                        help="Path to the experiment directory")
    parser.add_argument("-dd", "--data_dir", type=str,
                        default="/media/veracrypt1/MigrAVEProcessed/MigrAVEDaten", help="Path to the data directory")
    parser.add_argument("-od", "--output_dir", type=str, default="/media/veracrypt1/MigrAVEProcessed",
                        help="Path to the output directory")
    parser.add_argument("-m", "--modalities", required=True, type=str, nargs="+", help="List of modalities")
    parser.add_argument("-d", "--datasets", required=True, type=str, nargs="+", help="List of datasets")
    parser.add_argument("-pi", "--participant_ids", required=True, type=int, nargs="+", help="List of participant IDs")
    parser.add_argument("-s", "--sessions", type=int, nargs="+", help="List of sessions")
    parser.add_argument("-cn", "--classifier_name", required=True, type=str, help="Classifier name")
    args = parser.parse_args()
    # args = Args
    for parsed_dir in [args.experiment_dir, args.data_dir, args.output_dir]:
        if not Path(parsed_dir).is_dir():
            print(f"Parsed directory {parsed_dir} does not exist.")
            sys.exit(0)
    generate_prediction_video(experiment_dir=args.experiment_dir, data_dir=args.data_dir, output_dir=args.output_dir,
                              modalities=args.modalities, datasets=args.datasets, participant_ids=args.participant_ids,
                              sessions=args.sessions, classifier_name=args.classifier_name)

    #get_xgboost_cv_feature_importance(experiment_dir=args.experiment_dir, modalities=args.modalities, datasets=args.datasets, participant_ids=args.participant_ids)
