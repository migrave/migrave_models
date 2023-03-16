from typing import Union

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import sklearn.ensemble as ensemble
import sklearn.naive_bayes as naive_bayes
import sklearn.calibration as calibration
import sklearn.svm as svm
import sklearn.linear_model as linear_model
import sklearn.neural_network as neural_network
import xgboost
from tensorflow import keras
from hmmlearn.hmm import GaussianHMM
from sklearn_crfsuite import CRF

ALLOWED_CLASSIFIERS = ['random_forest', 'xgboost', 'adaboost', 'svm',
                       'knn', 'naive_bayes', 'logistic_regression', "neural_network",
                       "recurrent_neural_network", "hmm", "crf"]

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
                                      eval_metric='logloss',
                                      early_stopping_rounds=10,
                                      subsample=0.8,
                                      colsample_bynode=0.8,
                                      # scale_pos_weight=1 TODO: add weights sum(negative instances) / sum(positive instances)
                                      )
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
        model.add(keras.layers.LSTM(100, return_sequences=True, activation="tanh", dropout=0.2, recurrent_dropout=0.2))
        model.add(keras.layers.Dense(1, activation="sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    elif "hmm" == model_name:
        model = GaussianHMM(n_components=2, algorithm="viterbi")
    elif "crf" == model_name:
        model = CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
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
        validation_data = []
        validation_labels = []
        for i, sequence in enumerate(train_data):
            idx = int(sequence.shape[0] * 0.9)
            train_data[i] = sequence[:idx, :]
            validation_data.append(sequence[idx:, :])
            validation_labels.append(train_labels[i][idx:, :])
            train_labels[i] = train_labels[i][:idx, :]
        train_data = keras.preprocessing.sequence.pad_sequences(train_data, padding="post", dtype="float32", value=0.0)
        train_labels = keras.preprocessing.sequence.pad_sequences(train_labels, padding="post", dtype="float32", value=0.0)
        validation_data = keras.preprocessing.sequence.pad_sequences(validation_data, padding="post", dtype="float32", value=0.0)
        validation_labels = keras.preprocessing.sequence.pad_sequences(validation_labels, padding="post", dtype="float32", value=0.0)
        callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10)
        classifier.fit(train_data, train_labels, epochs=200, batch_size=min(200, len(train_data)), validation_data=(validation_data, validation_labels), callbacks=[callback])
        test_data = keras.preprocessing.sequence.pad_sequences(test_data, padding="post", dtype="float32", value=0.0)
        scores_1 = classifier.predict(test_data)
        scores_1 = [score[0] for score_batch, test_labels_batch in zip(scores_1, test_labels) for score in score_batch[:len(test_labels_batch)]]
        scores_0 = [1 - score_1 for score_1 in scores_1]
        test_labels = np.concatenate(test_labels).flatten()
        predictions = [target_names[np.rint(sc)] for sc in scores_1]
    elif isinstance(classifier, GaussianHMM):
        train_sequence_len = [len(sequence) for sequence in train_data]
        train_data = np.concatenate(train_data)
        classifier.fit(train_data, train_sequence_len)
        scores_assign = classifier.predict_proba(train_data, train_sequence_len)
        train_labels = np.concatenate(train_labels).flatten().tolist()
        auroc_assign_cls = [metrics.roc_auc_score(train_labels, scores_assign[:, cls]) for cls in range(scores_assign.shape[1])]
        label_1 = np.argmax(auroc_assign_cls)
        label_0 = 1 - label_1
        test_sequence_len = [len(sequence) for sequence in test_data]
        test_data = np.concatenate(test_data)
        scores = classifier.predict_proba(test_data, test_sequence_len)
        scores = scores[:, [label_0, label_1]]
        scores_1 = scores[:, 1]
        scores_0 = scores[:, 0]
        test_labels = np.concatenate(test_labels).flatten()
        predictions = [target_names[np.argmax(sc)] for sc in scores]
        # TODO: save the label_0 and label_1 to use model for classification
    elif isinstance(classifier, CRF):
        train_data = [[dict(zip(map(str, list(range(len(timestep)))), timestep)) for timestep in sequence] for sequence in train_data]
        train_labels = [[str(timestep[0]) for timestep in sequence] for sequence in train_labels]
        classifier.fit(train_data, train_labels)
        test_data = [[dict(zip(map(str, list(range(len(timestep)))), timestep)) for timestep in sequence] for sequence in test_data]
        scores = classifier.predict_marginals(test_data)
        scores = np.array([[timestamp["0"], timestamp["1"]] for sequence in scores for timestamp in sequence])
        scores_1 = scores[:, 1]
        scores_0 = scores[:, 0]
        if any(np.isnan(scores_1)):
            return classifier, None
        test_labels = np.concatenate(test_labels).flatten()
        predictions = [target_names[np.argmax(sc)] for sc in scores]
    elif isinstance(classifier, xgboost.XGBClassifier):
        train_data, valid_data, train_labels, valid_labels = train_test_split(train_data, train_labels, test_size=0.1, shuffle=False)
        classifier.fit(train_data, train_labels, eval_set=[(valid_data, valid_labels)])
        scores = classifier.predict_proba(test_data)
        scores_1 = scores[:, 1]
        scores_0 = scores[:, 0]
        predictions = [target_names[np.argmax(sc)] for sc in scores]
    else:
        classifier.fit(train_data, train_labels)
        scores = classifier.predict_proba(test_data)
        scores_1 = scores[:, 1]
        scores_0 = scores[:, 0]
        predictions = [target_names[np.argmax(sc)] for sc in scores]

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
    result["C_ij(i=label,j=prediction)"]= confusion_mtx

    return classifier, result
