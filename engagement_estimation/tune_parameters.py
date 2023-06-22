from abc import ABC

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, GroupKFold
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from photonai.base import Hyperpipe, PipelineElement, Switch
from photonai.optimization import IntegerRange, FloatRange
from photonai.base import PhotonRegistry
import os
import pandas as pd
import xgboost

from utils import merge_datasets, remove_label_issues, NON_FEATURES_COLS, NAN_MAX_COLS

custom_element_root_folder = "./photon_custom_transformer"
registry = PhotonRegistry(custom_elements_folder=custom_element_root_folder)

# registry.register(photon_name="XGBoostClassifier",
#                   class_str="XGBoostClassifier.XGBoostClassifier",
#                   element_type="Transformer")

registry.activate()

# show information about the element
# registry.info("XGBoostClassifier")

modalities = ["video", "game"]
datasets = ["features_video_right.csv"]
exclude_feature_regex = "^op.*|^of_ts_success.*|^ros_skill.*|^ros_mistakes.*|ros_games_session|ros_ts_game_start|^ros_diff.*|ros_aptitude.*|^of_success_.*"
exclude_samples_regex = "^of_success_features.*"
label_issue_file = "exclude_op_of_sucess_ros_scalable_video_game_voting_features_video_left_features_video_right_features_video_color_xgboost_cross_validation_issues.csv"


def fill_nans(features):
    features_copy = features.copy()
    for col in features_copy.columns:
        if not col.startswith(tuple(NAN_MAX_COLS)):
            min_val = np.nanmin(features_copy[col])
            features_copy[col] = features_copy[col].fillna(min_val)
        else:
            max_val = np.nanmax(features_copy[col])
            features_copy[col] = features_copy[col].fillna(max_val)
    return features_copy


def get_n_estimators(xgb_classifier, features, labels, folds):
    xgtrain = xgboost.DMatrix(features, label=labels)
    cvresult = xgboost.cv(xgb_classifier.get_xgb_params(), xgtrain,
                          num_boost_round=xgb_classifier.get_params()["n_estimators"], folds=folds,
                          metrics="logloss", early_stopping_rounds=50)
    print(cvresult.shape[0])


# def get_label_issues_idx(label_issue_file: str, feature_df: pd.DataFrame):
#     label_issues_idx_df = feature_df.copy()[["participant", "session_num", "timestamp"]]
#     label_issue_file = os.path.join("dataset", label_issue_file)
#     label_issue_df = pd.read_csv(label_issue_file, index_col=0)
#     label_issue_df = label_issue_df.loc[label_issue_df["label_issues"] == True]
#     label_issues_idx_df.reset_index(drop=True, inplace=True)
#     label_issues_idx_df.reset_index(names="label_issue_idx", inplace=True)
#     label_issues_idx_df = label_issues_idx_df.merge(right=label_issue_df, on=["participant", "session_num", "timestamp"], how="inner")
#     label_issues_idx_df = label_issues_idx_df["label_issue_idx"].values
#
#     return label_issues_idx_df
#
#
# class LeaveOneGroupOutLabelIssue(LeaveOneGroupOut, ABC):
#     def split(self, X, y=None, groups=None, label_issues=None):
#         logo_splits = super().split(X, y, groups)
#         logo_splits_label_issues = [[np.setdiff1d(split[0], label_issues), split[1]] for split in logo_splits]
#         return logo_splits_label_issues


def f1_score_0(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None, zero_division=0)[0]


def recall_score_0(y_true, y_pred):
    return recall_score(y_true, y_pred, average=None, zero_division=0)[0]


def precision_score_0(y_true, y_pred):
    return precision_score(y_true, y_pred, average=None, zero_division=0)[0]


pipe = Hyperpipe("reg_lambda_2", project_folder="./logs/tuning",
                 optimizer="grid_search",
                 # optimizer_params={"n_configurations": 32},
                 metrics=["accuracy", "balanced_accuracy", "recall", "precision", "f1_score",
                          ("f1_score_0", f1_score_0),
                          ("recall_score_0", recall_score_0),
                          ("precision_score_0", precision_score_0)],
                 best_config_metric="f1_score_0",
                 outer_cv=LeaveOneGroupOut(),
                 inner_cv=GroupKFold(n_splits=5),
                 use_test_set=True,
                 calculate_metrics_across_folds=True,
                 nr_of_processes=1)

pipe += PipelineElement("MinMaxScaler")

# pipe += PipelineElement("ImbalancedDataTransformer",
#                         hyperparameters={"method_name": ["RandomOverSampler", "SMOTE", "SMOTETomek"]},
#                         test_disabled=True,
#                         )

pipe += PipelineElement("XGBoostClassifier",
                        hyperparameters={  # "n_estimators": IntegerRange(50, 500, step=10),
                            # "max_depth": IntegerRange(3, 10),
                            # "subsample": FloatRange(.5, .9, num=5),
                            # "colsample_bytree": FloatRange(.5, .9, num=5),
                            # "learning_rate": FloatRange(.01, .3),
                            # "min_child_weight": IntegerRange(1, 6),
                            # "reg_lambda": [1, 1.5, 2, 2.5, 3],
                            # "reg_alpha": [0, 1e-6, 1e-5, 1e-4, 1e-3],
                            # "gamma": FloatRange(.0, 1., num=11),
                        },
                        n_estimators=298,
                        max_depth=9,
                        min_child_weight=4,
                        gamma=.1,
                        subsample=.9,
                        colsample_bytree=.9,
                        reg_alpha=1e-5,
                        reg_lambda=1.5
                        )

dataset_files = [os.path.join("dataset", dataset) for dataset in datasets]
features, dataset_stems = merge_datasets(dataset_files=dataset_files, modalities=modalities,
                                         exclude_feature_regex=exclude_feature_regex,
                                         exclude_samples_regex=exclude_samples_regex,
                                         label_issue_file=label_issue_file)
# label_issue_idx = get_label_issues_idx(feature_df=features, label_issue_file="xgboost_cross_validation_issues.csv")
labels = features[["engagement"]].values.flatten()
groups = features[["participant"]].values.flatten()
features = fill_nans(features.drop(columns=NON_FEATURES_COLS)).values

y_unique, y_counts = np.unique(labels, return_counts=True)
n_0 = y_counts[np.argmin(y_unique)]
n_1 = y_counts[np.argmax(y_unique)]
scale_pos_weight = n_0 / n_1
xgb_classifier = xgboost.XGBClassifier(n_estimators=2000,
                                       max_depth=9,
                                       booster="gbtree",
                                       n_jobs=-1,
                                       eval_metric="logloss",
                                       subsample=.9,
                                       colsample_bytree=.9,
                                       use_label_encoder=False,
                                       learning_rate=.1,
                                       min_child_weight=4,
                                       reg_lambda=1.5,
                                       reg_alpha=1e-5,
                                       gamma=.1,
                                       scale_pos_weight=scale_pos_weight,
                                       tree_method="gpu_hist",
                                       gpu_id=0)
logo = LeaveOneGroupOut()
logo_splits = logo.split(features, labels, groups)

# uncomment to test for best number of estimators
# folds = [tuple([list(in_out) for in_out in fold]) for fold in logo_splits]
# get_n_estimators(xgb_classifier=xgb_classifier, features=features, labels=labels, folds=folds)

pipe.fit(features, labels, groups=groups)
