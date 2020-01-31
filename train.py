# !usr//bin/env python
# -*- coding:utf-8 -*-

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import xgboost as xgb

from scripts.preprocess import PreProcessor


class Model:
    @classmethod
    def naive_train_with_xgboost(cls, X_train, y_train):
        """XGBoostのデフォルトパラメータで推計


        """
        clf = xgb.XGBRegressor()
        clf.fit(X_train, y_train)
        return clf

    @classmethod
    def gridsearch_train_with_xgboost(cls, X_train, y_train):
        """XGBoostでグリッドサーチ

        best_prams:

        """
        params = {
            'max_depth': [5, 6, 7],
            'subsample': [0.9, 1.0],
            'learning_rate': [0.1, 0.2, 0.3],
            # 'min_child_weight': [0.9, 1, 10],
        }
        xgboost = xgb.XGBClassifier()
        clf = GridSearchCV(xgboost, params, scoring='neg_mean_absolute_error')
        clf.fit(X_train, y_train)
        print(clf.best_estimator_)

        return clf

    @classmethod
    def naive_train_with_svm(cls, X_train, y_train):
        """デフォルトパラメータのSVM

        """
        clf = SVR()
        clf.fit(X_train, y_train)
        return clf

    @classmethod
    def gridsearch_train_with_svr(cls, X_train, y_train):
        """グリッドサーチのSVM


        """
        params = {
            'C': [1, 10, 100],
            'gamma': [1.0, 0.1, 0.01],
        }
        svr = SVR()
        clf = GridSearchCV(svr, params, scoring='neg_mean_absolute_error')
        clf.fit(X_train, y_train)
        print('best_estimator:', clf.best_estimator_)
        return clf

    @classmethod
    def evaluate(cls, clf, X_test, y_test):
        """Evaluate with MAE.
        """
        y_pred = clf.predict(X_test)
        score = mean_absolute_error(y_test, y_pred)
        print('accuracy_score:', score)
        return score

    @classmethod
    def main(cls):
        X_train, X_test, y_train, y_test = PreProcessor.get_train_data('A')

        # naive_xgboost_clf = naive_train_with_xgboost(X_train, y_train)
        # evaluate(naive_xgboost_clf, X_test, y_test)
        #
        gridsearch_xgboost_clf = cls.gridsearch_train_with_xgboost(X_train, y_train)
        cls.evaluate(gridsearch_xgboost_clf, X_test, y_test)

        # naive_svm_clf = naive_train_with_svm(X_train, y_train)
        # evaluate(naive_svm_clf, X_test, y_test)

        # gridsearch_svm_clf = gridsearch_train_with_svm(X_train, y_train)
        # evaluate(gridsearch_svm_clf, X_test, y_test)


if __name__ == '__main__':
    Model.main()
