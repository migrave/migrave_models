import numpy as np
from sklearn import metrics


def sklearn(train_data,
            train_labels,
            test_data,
            test_labels,
            classifier,
            target_names={0:-1, 1:0, 2:1}):
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
    predictions = [target_names[np.argmax(sc)] for sc in scores]

    # classification report
    cls_report = metrics.classification_report(test_labels,
                                               predictions,
                                               target_names=list(target_names.values()),
                                               output_dict=True)
    auroc = metrics.roc_auc_score(test_labels, scores, multi_class="ovr")

    result = {}
    result["AUROC"] = auroc
    for i,cls in enumerate(cls_report.keys()):
        if cls in target_names.values():
            result[f"Precision_{cls}"] = cls_report[cls]["precision"]
            result[f"Recall_{cls}"] = cls_report[cls]["recall"]
            result[f"F1_{cls}"] = cls_report[cls]["f1-score"]
        elif cls == "accuracy":
            result["Accuracy"] = cls_report[cls]

    return classifier, result
