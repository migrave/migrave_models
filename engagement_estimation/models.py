from sklearn import metrics

def sklearn(train_data,
            train_labels,
            test_data,
            test_labels,
            classifier):
    """
    Train generalized model: leave one out for test
    """

    classifier.fit(train_data.values, train_labels.values)

    scores = classifier.predict_proba(test_data.values)[:,1]
    pred = [round(value) for value in scores]   
    accuracy = metrics.accuracy_score(test_labels, pred)

    result = {}
    result["Accuracy"] = accuracy

    auroc = metrics.roc_auc_score(test_labels, scores)
    result["AUROC"] = auroc

    #set average to None to computer per-class precision
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(test_labels, 
                                                                       pred, 
                                                                       average=None, 
                                                                       labels=[0,1])

    result['Precision_0'], result['Precision_1'] = precision[0], precision[1]
    result['Recall_0'], result['Recall_1'] = recall[0], recall[1]
    result['F1_0'], result['F1_1'] = f1[0], f1[1]

    return classifier, result
