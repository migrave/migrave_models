from typing import Union

import numpy as np
from sklearn import metrics
from sklearn import neighbors
import sklearn.ensemble as ensemble
import sklearn.naive_bayes as naive_bayes
import sklearn.calibration as calibration
import sklearn.svm as svm
import sklearn.linear_model as linear_model
from sklearn.neural_network import MLPClassifier
import xgboost

ALLOWED_CLASSIFIERS = ['random_forest', 'xgboost', 'adaboost', 'svm',
                       'knn', 'naive_bayes', 'logistic_regression', "neural_network"]

def get_classifier(model_name: str, feature_n: int) -> Union[ensemble.RandomForestClassifier,
                                             xgboost.XGBClassifier,
                                             ensemble.AdaBoostClassifier,
                                             calibration.CalibratedClassifierCV,
                                             neighbors.KNeighborsClassifier,
                                             naive_bayes.GaussianNB,
                                             linear_model.LogisticRegression,
                                             keras.Sequential]:
    """Returns a scikit-learn classifier object corresponding to the given model name.
    The following classifier names are allowed:
        random_forest, xgboost, adaboost, svm, knn, naive_bayes, and logistic_regression.
    Raises a ValueError if a non-supported model name is specified.

    If 'random_forest' is passed, uses 100 trees.
    In the case of 'adaboost', uses a random forest with 100 trees.
    For 'xgboost', 100 estimators with maximum depth of 6 are used.
    In th case of 'naive_bayes', a Gaussian naive Bayes classifier is used.

    Keyword arguments:
    @param model_name: str -- name of the classifier to be instantiated

    """
    if model_name not in ALLOWED_CLASSIFIERS:
        raise ValueError(f"Classifier {model_name} is not supported")

    model = None
    if "random_forest" == model_name:
        model = ensemble.RandomForestClassifier(n_estimators=100,
                                                max_depth=None,
                                                max_features=None,
                                                n_jobs=-1)
    elif "xgboost" == model_name:
        model = xgboost.XGBClassifier(n_estimators=100,
                                      max_depth=6,
                                      booster='gbtree',
                                      n_jobs=-1,
                                      eval_metric='logloss')
    elif "adaboost" == model_name:
        model = ensemble.AdaBoostClassifier(ensemble.RandomForestClassifier(n_estimators=100))
    elif "svm" == model_name:
        model = calibration.CalibratedClassifierCV(svm.LinearSVC())
    elif "knn" == model_name:
        model = neighbors.KNeighborsClassifier(n_neighbors=5)
    elif "naive_bayes" == model_name:
        model = naive_bayes.GaussianNB()
    elif "logistic_regression" == model_name:
        model = linear_model.LogisticRegression(penalty='l2', solver='liblinear')
    elif "neural_network" == model_name:
        model = MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", early_stopping=True)
    return model

def sklearn(train_data,
            train_labels,
            test_data,
            test_labels,
            classifier,
            target_names={0:0, 1:1}):
    """
    Train classifier
    Input:
      train_data: train data
      train_labels: train labels
      test_data: test data
      test_labels: test labels
      target_names: an index mapping to labels
    Return:
      Classifier and dictionary containing the results
    """

    classifier.fit(train_data, train_labels)
    scores = classifier.predict_proba(test_data)
    scores_1 = scores[:, 1]
    predictions = [target_names[np.argmax(sc)] for sc in scores]

    # classification report
    cls_report = metrics.classification_report(test_labels,
                                               predictions,
                                               target_names=list(target_names.values()),
                                               output_dict=True)
    auroc = metrics.roc_auc_score(test_labels, scores_1, multi_class="ovr")

    result = {}
    result["AUROC"] = auroc
    for cls in cls_report.keys():
        if cls in target_names.values():
            result[f"Precision_{cls}"] = cls_report[cls]["precision"]
            result[f"Recall_{cls}"] = cls_report[cls]["recall"]
            result[f"F1_{cls}"] = cls_report[cls]["f1-score"]
        elif cls == "accuracy":
            result["Accuracy"] = cls_report[cls]

    return classifier, result
