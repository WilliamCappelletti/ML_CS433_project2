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

#-------------------------------------------------------------------------------
# MODELS
#------------------------------------------------------------------------------
# XGBoost IMPLEMENTATION

param = {'max_depth': 10, 'eta': 1, 'silent': 1, 'subsample': 0.8, #default parameters for easyXGB constuctor
 'reg_alpha': 0.7,  'tree_method': 'auto'}

class easyXGB :
    '''Wrapper to use XGBooster models with sklearn methods.
    Implement fit, predict and score.
    '''

    def __init__(self, load=0, params_=param, scorer = 'MSE'):
        self.params = {} if (params_ is None) else params_
        self._scorer = scorer
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
        elif self._scorer == 'R2':
            return explained_variance_score(y_test, y_pred)
        else :
            raise NotImplementedError

    def save(self, number):
        if number in range(1,5):
            self.model.save_model('reg_{i}.model'.format(i=number))
        else :
            raise NotImplementedError

class XGBImplement:
    '''Model that include easyXGB implentation for the 4 responses.
    Methods:
    predict(self, X_test)  Predict values for the log of 'k1_bwd_effective','k1_fwd_effective', 'k2_bwd_effective','k2_fwd_effective' using our model.

    Variables:
    scores                 Contains scores on test and train predictions as a pandas DataFrame.
    '''
    def __init__(self, random_state = 33, realization_split = False):
        y1, y2, y3, y4, X, realization = treatment.data_initialization(realiz=True)
        X = X.drop(columns=['sigma_mass_0.0', 'sigma_mass_0.825', 'enzyme_concentration'])
        X = X.values

        self.k1_bkw_model = easyXGB(params_= {'reg_alpha': 0.33684210526315794, 'eta': 0.7, 'max_depth': 9, 'subsample': 1.0}, scorer='R2')
        self.k1_fwd_model = easyXGB(params_= {'reg_alpha': 0.9, 'eta': 0.6, 'max_depth': 4, 'subsample': 1.0}, scorer='R2')
        self.k2_bkw_model = easyXGB(params_= {'reg_alpha': 0.95, 'eta': 0.65, 'max_depth': 4, 'subsample': 1.0}, scorer='R2')
        self.k2_fwd_model = easyXGB(params_= {'reg_alpha': 0.4, 'eta': 0.66, 'max_depth': 10, 'subsample': 1.0}, scorer='R2')

        indices = ['k1_bwd_effective','k1_fwd_effective', 'k2_bwd_effective','k2_fwd_effective']
        self.scores = pd.DataFrame({'R^2 train score': 4*[0], 'R^2 test score': 4*[0],
                                'MSE train score': 4*[0], 'MSE test score': 4*[0]},
                                index = indices)

        iterator = [(self.k1_bkw_model, y1, indices[0]), (self.k1_fwd_model, y2, indices[1]),
                    (self.k2_bkw_model, y3, indices[2]), (self.k2_fwd_model, y4, indices[3])]
        for model, y, index in iterator:
            X_train, X_test, y_train, y_test = treatment.train_test_split(X, y, test_size=0.4, random_state = random_state) if not realization_split else treatment.train_test_split_realiz(X, y, realization, test_size=0.4, random_state = random_state)
            model.fit(X_train, y_train)
            y_pred_tr = model.predict(X_train)
            y_pred_te = model.predict(X_test)

            #['R^2 train score', 'R^2 test score', 'MSE train score', 'MSE test score']
            self.scores.loc[index] = model.score(X_train, y_train), model.score(X_test, y_test), mean_squared_error(y_train, y_pred_tr),  mean_squared_error(y_test, y_pred_te)

    def predict(self, X_test):
        ''' Predict values for the log of 'k1_bwd_effective','k1_fwd_effective', 'k2_bwd_effective','k2_fwd_effective' using our model.

        Returns pandas DataFrame with columns k1_bwd_pred, k1_fwd_pred, k2_bwd_pred, k2_fwd_pred (logscaled)
        '''
        X = X_test.drop(columns=['sigma_mass_0.0', 'sigma_mass_0.825', 'enzyme_concentration'])
        X = X.values

        y1 = self.k1_bkw_model.predict(X)
        y2 = self.k1_fwd_model.predict(X)
        y3 = self.k2_bkw_model.predict(X)
        y4 = self.k2_fwd_model.predict(X)

        Y_pred = pd.DataFrame({'k1_bwd_pred': y1, 'k1_fwd_pred': y2,
                                'k2_bwd_pred': y3, 'k2_fwd_pred': y4})
        return Y_pred


#-------------------------------------------------------------------------------
# Ridge
class RidgeImplement:
    '''Wrapper of sklearn RidgeCV for our purpose.

    Methods:
    predict(self, X_test)  Predict values for the log of 'k1_bwd_effective','k1_fwd_effective', 'k2_bwd_effective','k2_fwd_effective' using our model.

    Variables:
    scores                 Contains scores on test and train predictions as a pandas DataFrame.
    '''

    def __init__(self, degree=2, interaction_only=False, random_state = 7, realization_split = False):
        y1, y2, y3, y4, X, realization = treatment.data_initialization(realiz=True)
        X = treatment.polynomial_data(X, degree, interaction_only, categories=True)

        self.degree = degree
        self.interaction_only = interaction_only
        self.k1_bkw_model = RidgeCV(alphas=np.logspace(-5,5,40), cv=None, fit_intercept=False)
        self.k1_fwd_model = RidgeCV(alphas=np.logspace(-5,5,60), cv=None, fit_intercept=False)
        self.k2_bkw_model = RidgeCV(alphas=np.logspace(-5,5,60), cv=None, fit_intercept=False)
        self.k2_fwd_model = RidgeCV(alphas=np.logspace(-5,5,40), cv=None, fit_intercept=False)

        indices = ['k1_bwd_effective','k1_fwd_effective', 'k2_bwd_effective','k2_fwd_effective']
        self.scores = pd.DataFrame({'R^2 train score': 4*[0], 'R^2 test score': 4*[0],
                                'MSE train score': 4*[0], 'MSE test score': 4*[0]},
                                index = indices)

        iterator = [(self.k1_bkw_model, y1, indices[0]), (self.k1_fwd_model, y2, indices[1]),
                    (self.k2_bkw_model, y3, indices[2]), (self.k2_fwd_model, y4, indices[3])]
        for model, y, index in iterator:
            X_train, X_test, y_train, y_test = treatment.train_test_split(X, y, test_size=0.4, random_state = random_state) if not realization_split else treatment.train_test_split_realiz(X, y, realization, test_size=0.4, random_state = random_state)
            model.fit(X_train, y_train)
            y_pred_tr = model.predict(X_train)
            y_pred_te = model.predict(X_test)

            #['R^2 train score', 'R^2 test score', 'MSE train score', 'MSE test score']
            self.scores.loc[index] = model.score(X_train, y_train), model.score(X_test, y_test), mean_squared_error(y_train, y_pred_tr),  mean_squared_error(y_test, y_pred_te)

    def predict(self, X_test):
        ''' Predict values for the log of 'k1_bwd_effective','k1_fwd_effective', 'k2_bwd_effective','k2_fwd_effective' using our model.

        Returns pandas DataFrame with columns k1_bwd_pred, k1_fwd_pred, k2_bwd_pred, k2_fwd_pred (logscaled)
        '''
        X = treatment.polynomial_data(X_test, self.degree, self.interaction_only, categories=True)
        y1 = self.k1_bkw_model.predict(X)
        y2 = self.k1_fwd_model.predict(X)
        y3 = self.k2_bkw_model.predict(X)
        y4 = self.k2_fwd_model.predict(X)

        Y_pred = pd.DataFrame({'k1_bwd_pred': y1, 'k1_fwd_pred': y2,
                                'k2_bwd_pred': y3, 'k2_fwd_pred': y4})
        return Y_pred

#-------------------------------------------------------------------------------
# Support vector regression

class SVRImplement:
    '''Wrapper of sklearn Support vector regression for our purpose.

    Methods:
    predict(self, X_test)  Predict values for the log of 'k1_bwd_effective','k1_fwd_effective', 'k2_bwd_effective','k2_fwd_effective' using our model.

    Variables:
    scores                 Contains scores on test and train predictions as a pandas DataFrame.
    '''

    def __init__(self, random_state = 5, realization_split = False):
        y1, y2, y3, y4, X, realization = treatment.data_initialization(realiz=True)
        X = treatment.polynomial_data(X, 2, interaction_only=True, categories=True)


        self.k1_bkw_model = SVR(kernel='rbf', max_iter=5*10**4, cache_size=3000)
        self.k1_fwd_model = SVR(kernel='rbf', max_iter=5*10**4, cache_size=3000)
        self.k2_bkw_model = SVR(kernel='rbf', max_iter=5*10**4, cache_size=3000)
        self.k2_fwd_model = SVR(kernel='rbf', max_iter=5*10**4, cache_size=3000)

        indices = ['k1_bwd_effective','k1_fwd_effective', 'k2_bwd_effective','k2_fwd_effective']
        self.scores = pd.DataFrame({'R^2 train score': 4*[0], 'R^2 test score': 4*[0],
                                'MSE train score': 4*[0], 'MSE test score': 4*[0]},
                                index = indices)

        iterator = [(self.k1_bkw_model, y1, indices[0]), (self.k1_fwd_model, y2, indices[1]),
                    (self.k2_bkw_model, y3, indices[2]), (self.k2_fwd_model, y4, indices[3])]
        for model, y, index in iterator:
            X_train, X_test, y_train, y_test = treatment.train_test_split(X, y, test_size=0.4, random_state = random_state) if not realization_split else treatment.train_test_split_realiz(X, y, realization, test_size=0.4, random_state = random_state)
            model.fit(X_train, y_train)
            y_pred_tr = model.predict(X_train)
            y_pred_te = model.predict(X_test)

            #['R^2 train score', 'R^2 test score', 'MSE train score', 'MSE test score']
            self.scores.loc[index] = model.score(X_train, y_train), model.score(X_test, y_test), mean_squared_error(y_train, y_pred_tr),  mean_squared_error(y_test, y_pred_te)

    def predict(self, X_test):
        ''' Predict values for the log of 'k1_bwd_effective','k1_fwd_effective', 'k2_bwd_effective','k2_fwd_effective' using our model.

        Returns pandas DataFrame with columns k1_bwd_pred, k1_fwd_pred, k2_bwd_pred, k2_fwd_pred (logscaled)
        '''
        X = treatment.polynomial_data(X_test, 2, categories=True)
        y1 = self.k1_bkw_model.predict(X)
        y2 = self.k1_fwd_model.predict(X)
        y3 = self.k2_bkw_model.predict(X)
        y4 = self.k2_fwd_model.predict(X)

        Y_pred = pd.DataFrame({'k1_bwd_pred': y1, 'k1_fwd_pred': y2,
                                'k2_bwd_pred': y3, 'k2_fwd_pred': y4})
        return Y_pred
#-------------------------------------------------------------------------------
# REGRESSIONS

def reproduction_ridge(csv = False, degree=2, interaction_only=False, random_state = 7, realization_split = False):
    '''Ridge reproduction.

    This function reproduces the various ridge regressions prosented in paper,
    fitting a CV ridge regression for the logarithms of 'k1_bwd_effective','k1_fwd_effective',
    'k2_bwd_effective','k2_fwd_effective', as explained in the paper.

    Returns the model.

    Arguments:
    -csv                    True by default. If true the function output the predictions in a
                            .csv file.
    -random_state:          7 by default (used in regressions).
    -realization_split:     False by default. If False performs usual traint-test split, if True
                            performs train-test split realization-wise.
    '''

    #y1, y2, y3, y4, X = treatment.data_initialization()

    fitted_model = RidgeImplement(degree, interaction_only, random_state, realization_split)

    if csv:
        _, _, _, _, X = treatment.data_initialization()
        Y_pred = fitted_model.predict(X)
        Y_pred.to_csv('../results/Ridge_reproduction.csv')

    return fitted_model

def reproduction_svr(csv = False, random_state = 5, realization_split = False):
    '''Support Vector Regression reproduction.

    This function reproduces the SVR prosented in paper,
    fitting four models for the logarithms of 'k1_bwd_effective','k1_fwd_effective',
    'k2_bwd_effective','k2_fwd_effective', as explained in the paper.

    Returns the model.

    Arguments:
    -csv                    True by default. If true the function output the predictions in a
                            .csv file.
    -random_state:          5 by default (used in regressions).
    -realization_split:     False by default. If False performs usual traint-test split, if True
                            performs train-test split realization-wise.
    '''

    #y1, y2, y3, y4, X = treatment.data_initialization()

    fitted_model = SVRImplement(random_state, realization_split)

    if csv:
        _, _, _, _, X = treatment.data_initialization()
        Y_pred = fitted_model.predict(X)
        Y_pred.to_csv('../results/SVR_reproduction.csv')

    return fitted_model

def reproduction_XGBoost(csv = False, random_state = 33, realization_split = False):
    '''XGBoost reproduction.

    This function reproduces the best XGBoost regressions obtained,
    fitting four models for the logarithms of 'k1_bwd_effective','k1_fwd_effective',
    'k2_bwd_effective','k2_fwd_effective', as explained in the paper.

    Returns the model.

    Arguments:
    -csv                    True by default. If true the function output the predictions in a
                            .csv file.
    -random_state:          33 by default (used in regressions).
    -realization_split:     False by default. If False performs usual traint-test split, if True
                            performs train-test split realization-wise.
    '''

    #y1, y2, y3, y4, X = treatment.data_initialization()

    fitted_model = XGBImplement(random_state, realization_split)

    if csv:
        _, _, _, _, X = treatment.data_initialization()
        Y_pred = fitted_model.predict(X)
        Y_pred.to_csv('../results/XGB_reproduction.csv')

    return fitted_model
