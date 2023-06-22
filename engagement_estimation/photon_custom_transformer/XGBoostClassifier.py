from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost
import numpy as np


class XGBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=6, subsample=.8, colsample_bytree=.8, learning_rate=.3, min_child_weight=1., reg_lambda=1., reg_alpha=.0, gamma=.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma

        self.model = None

    def fit(self, X, y, targets=None, **kwargs):
        y_unique, y_counts = np.unique(y, return_counts=True)
        n_0 = y_counts[np.argmin(y_unique)]
        n_1 = y_counts[np.argmax(y_unique)]
        scale_pos_weight = n_0 / n_1
        self.model = xgboost.XGBClassifier(n_estimators=self.n_estimators,
                                           max_depth=self.max_depth,
                                           booster="gbtree",
                                           n_jobs=-1,
                                           eval_metric="logloss",
                                           subsample=self.subsample,
                                           colsample_bytree=self.colsample_bytree,
                                           use_label_encoder=False,
                                           learning_rate=self.learning_rate,
                                           min_child_weight=self.min_child_weight,
                                           reg_lambda=self.reg_lambda,
                                           reg_alpha=self.reg_alpha,
                                           gamma=self.gamma,
                                           scale_pos_weight=scale_pos_weight,
                                           tree_method="gpu_hist",
                                           gpu_id=0)
        # X, X_valid, y, y_valid = train_test_split(X, y, test_size=.1, stratify=y)
        self.model.fit(X, y)  #, eval_set=[(X_valid, y_valid)], early_stopping_rounds=50)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def transform(self, data, targets=None, **kwargs):
        return data
