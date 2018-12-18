# -*- coding: utf-8 -*-
'''Models application.'''

import treatment
# from cross_validation import multi_cross_validation

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.svm import SVR
#------------------------------------------------------------------------------
# XGBoost IMPLEMENTATION

param = {'max_depth': 10, 'eta': 1, 'silent': 1, 'subsample': 0.8, #default parameters for easyXGB constuctor
 'reg_alpha': 0.7,  'tree_method': 'auto'}

class easyXGB :
    '''Wrapper to use XGBooster models with sklearn methods.
    Implement fit, predict and score.
    '''

    def __init__(self, load=0, params_=param):
        self.params = {} if (params_ is None) else params_
        self._scorer = 'MSE'
        if load in range(1,5):
            self.model = xgb.Booster()
            self.model.load_model('reg_{i}.model'.format(i=load))

    def set_params(self, **kwargs):
        for name in kwargs:
            self.params[name] = kwargs[name]

    def fit(self, X_train, y_train, **parameters):
        '''Wrapper for xgb fit function

        Fit the easyXGB model to y_train and X_train.'''

        self.set_params(**parameters)

        dtrain = xgb.DMatrix(X_train, label=y_train)

        self.model = xgb.train(params=self.params, dtrain=dtrain)

    def predict(self, X_test):
        '''Wrapper for xgb prediction function

        Returns predited values for X_test observations.'''

        dtest = xgb.DMatrix(X_test)

        return self.model.predict(dtest)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        if self._scorer == 'MSE':
            return mean_squared_error(y_test, y_pred)
        else:
             raise NotImplementedError

    def save(self, number):
        if number in range(1,5):
            self.model.save_model('reg_{i}.model'.format(i=number))
        else :
            raise NotImplementedError
