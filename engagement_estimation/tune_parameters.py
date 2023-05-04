import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from photonai.base import Hyperpipe, PipelineElement, Switch
from photonai.optimization import IntegerRange, FloatRange
from photonai.base import PhotonRegistry
import os

from utils import merge_datasets, NON_FEATURES_COLS, NAN_MAX_COLS

custom_element_root_folder = "./photon_custom_transformer"
registry = PhotonRegistry(custom_elements_folder=custom_element_root_folder)

# registry.register(photon_name="XGBoostClassifier",
#                   class_str="XGBoostClassifier.XGBoostClassifier",
#                   element_type="Transformer")

registry.activate()

# show information about the element
# registry.info("XGBoostClassifier")

modalities = ["video", "audio", "game"]
datasets = ["features_video_right.csv", "features_video_left.csv", "features_video_color.csv"]


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


def f1_score_0(y_true, y_pred):
    return f1_score(y_true, y_pred, average=None, zero_division=0)[0]


def recall_score_0(y_true, y_pred):
    return recall_score(y_true, y_pred, average=None, zero_division=0)[0]


def precision_score_0(y_true, y_pred):
    return precision_score(y_true, y_pred, average=None, zero_division=0)[0]


pipe = Hyperpipe("migrave_pipe", project_folder="./logs/tuning",
                 optimizer="random_grid_search",
                 optimizer_params={"n_configurations": 50},
                 metrics=["accuracy", "recall", "precision", "f1_score",
                          ("f1_score_0", f1_score_0),
                          ("recall_score_0", recall_score_0),
                          ("precision_score_0", precision_score_0)],
                 best_config_metric="f1_score_0",
                 outer_cv=LeaveOneGroupOut(),
                 inner_cv=StratifiedKFold(n_splits=10),
                 use_test_set=True)

pipe += Switch("StandardizationSwitch",
               [PipelineElement("StandardScaler"), PipelineElement("MinMaxScaler")])

pipe += PipelineElement("ImbalancedDataTransformer",
                        hyperparameters={"method_name": ["RandomUnderSampler", "RandomOverSampler", "SMOTE"]})

pipe += PipelineElement("XGBoostClassifier",
                        hyperparameters={"n_estimators": IntegerRange(50, 500, step=10),
                                         "max_depth": IntegerRange(3, 10, step=1),
                                         "subsample": FloatRange(.5, 1., num=6),
                                         "colsample_bytree": IntegerRange(.5, 1., num=6),
                                         "learning_rate": FloatRange(.01, .3, num=16),
                                         "min_child_weight": IntegerRange(1, 6, step=1),
                                         "reg_lambda": FloatRange(.0, 5., num=11),
                                         "reg_alpha": FloatRange(.0, 5., num=11),
                                         "gamma": FloatRange(.0, 1., num=11)})

dataset_files = [os.path.join("dataset", dataset) for dataset in datasets]
features, dataset_stems = merge_datasets(dataset_files, modalities)
labels = features[["engagement"]].values.flatten()
groups = features[["participant"]].values.flatten()
features = fill_nans(features.drop(columns=NON_FEATURES_COLS)).values

pipe.fit(features, labels, groups=groups)
