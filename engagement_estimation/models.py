from typing import Union

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import neighbors
import sklearn.ensemble as ensemble
import sklearn.naive_bayes as naive_bayes
import sklearn.calibration as calibration
import sklearn.svm as svm
import sklearn.linear_model as linear_model
import sklearn.neural_network as neural_network
import xgboost
from tensorflow import keras

ALLOWED_CLASSIFIERS = ['random_forest', 'xgboost', 'adaboost', 'svm',
                       'knn', 'naive_bayes', 'logistic_regression', "neural_network",
                       "recurrent_neural_network"]

SEQUENTIAL_CLASSIFIERS = ["recurrent_neural_network", "hmm", "crf"]


def get_classifier(model_name: str) -> Union[ensemble.RandomForestClassifier,
                                             xgboost.XGBClassifier,
                                             ensemble.AdaBoostClassifier,
                                             calibration.CalibratedClassifierCV,
                                             neighbors.KNeighborsClassifier,
                                             naive_bayes.GaussianNB,
                                             linear_model.LogisticRegression,
                                             neural_network.MLPClassifier]:
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
        model = neural_network.MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam", early_stopping=True)
    elif "recurrent_neural_network" == model_name:
        model = keras.Sequential()
        model.add(keras.layers.Masking(mask_value=0.0))
        model.add(keras.layers.LSTM(100, return_sequences=True, activation="tanh"))
        model.add(keras.layers.Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
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
    if isinstance(classifier, keras.Sequential):
        callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10)
        classifier.fit(train_data, train_labels, epochs=200, batch_size=min(200, len(train_data)), validation_split=.1, callbacks=[callback])
        test_data = keras.preprocessing.sequence.pad_sequences(test_data, padding="post", dtype="float32", value=0.0)
        scores_1 = classifier.predict(test_data)
        scores_1 = [score[0] for score_batch, test_labels_batch in zip(scores_1, test_labels) for score in score_batch[:len(test_labels_batch)]]
        test_labels = np.concatenate(test_labels).flatten().tolist()
        predictions = [target_names[np.rint(sc)] for sc in scores_1]
    else:
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
