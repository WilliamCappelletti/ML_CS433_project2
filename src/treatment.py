# -*- coding: utf-8 -*-
'''This module contains general functions that can be used to pretreat
 and extract the 'result_full_factorial_pgm' dataset, alongside functions that
 perform the regressions dicussed in the paper.'''

import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#------------------------------------------------------------------------------
# DATA EXTRACTION AND TREATMENT

def cast_categories(frame, column, cat_name = 'Group'):
    '''Returns a new data frame in which the specified column has been changed by new colums representing the appertenance to one specified category'''

    new_frame = frame.copy()
    categories = new_frame[column].unique()
    splitted_cols = {'{cat_name}_{x}'.format(cat_name=cat_name, x=x): [ 1 if inside else 0 for inside in new_frame[column]==x] for x in categories}
    splitted_cols = pd.DataFrame(splitted_cols, index = new_frame.index)
    new_frame = pd.concat([new_frame.drop(columns=[column]), splitted_cols], axis=1)
    return new_frame

def data_initialization(realiz = False):
    '''Read and split data into response and features, with the optional list of realizations (simulation number).

    return y1, y2, y3, y4, X[, realization]'''
    # define data path
    data_folder = '../Data/'

    # read the data
    data = pd.read_csv(data_folder +'result_full_factorial_pgm.zip', index_col = 0)
    data.drop('Unnamed: 0.1', axis=1, inplace=True)

    y1, y2, y3, y4 = np.log(data['k1_bwd_effective']), np.log(data['k1_fwd_effective']), np.log(data['k2_bwd_effective']), np.log(data['k2_fwd_effective'])
    realization = data['realization']

    X = data.drop(columns=['k1_bwd_effective','k1_fwd_effective','k2_bwd_effective','k2_fwd_effective',
                      'k1_bwd_relative','k1_fwd_relative','k2_bwd_relative','k2_fwd_relative', 'realization'])

    X = cast_categories(X, 'sigma_mass', 'sigma_mass')

    X[['enzyme_complex_concentration', 'enzyme_concentration', 'product_concentration',
       'substrate_concentration']] = X[['enzyme_complex_concentration', 'enzyme_concentration', 'product_concentration',
       'substrate_concentration']].apply(lambda x : np.log(x))

    if realiz :
        return y1, y2, y3, y4, X, realization
    return y1, y2, y3, y4, X

def polynomial_data(X, deg=2, interaction_only=False, categories=True):
    '''This function cast polynomial expansion for the 'result_full_factorial_pgm' dataset.
    Built around sklearn.PolynomialFeatures.fit_transform().

    Returns a new numpy.array'''

    X1 = X[['enzyme_complex_concentration', 'enzyme_concentration', 'mu_mass', 'product_concentration',
       'substrate_concentration', 'volume_fraction']].values
    X2 = X[['sigma_mass_0.0', 'sigma_mass_0.825']].values

    interaction_poly = PolynomialFeatures(deg-1, interaction_only=interaction_only, include_bias=False)
    X3 = interaction_poly.fit_transform(X1) if deg > 1 else X1
    X3 = np.concatenate((X2[:,0].reshape(-1,1) * X3, X2[:,1].reshape(-1,1) * X3), axis=1)

    poly = PolynomialFeatures(deg, interaction_only=interaction_only, include_bias=False)
    X1 = poly.fit_transform(X1)

    return np.concatenate((X1, X3, X2) if categories else (X1,X2), axis = 1)

def train_test_split_realiz(X, Y, realization, **options):
    '''
    Function built around sklearn.train_test_split to split result_full_factorial_pgm data inside the same 'realization'

    Takes the same options as sklearn function.
    '''
    R = cast_categories(realization.to_frame(), 'realization', 're')
    X_, Y_ = X[R['re_0.0'].values == 1], Y[R['re_0.0'].values == 1]
    X_train, X_test, Y_train, Y_test = train_test_split(X_, Y_, **options)
    for i in range(1, R.shape[1]) :
        X_, Y_ = X[R['re_{i}.0'.format(i=i)].values == 1], Y[R['re_{i}.0'.format(i=i)].values == 1]
        X_train_tmp, X_test_tmp, Y_train_tmp, Y_test_tmp = train_test_split(X_, Y_, **options)
        X_train = np.concatenate((X_train, X_train_tmp))
        X_test = np.concatenate((X_test, X_test_tmp))
        Y_train = np.concatenate((Y_train, Y_train_tmp))
        Y_test = np.concatenate((Y_test, Y_test_tmp))

    if 'random_state' in options :
        np.random.seed(seed=options['random_state'])

    shuffledIndices_tr = np.random.permutation(X_train.shape[0])
    shuffledIndices_te = np.random.permutation(X_test.shape[0])
    return X_train[shuffledIndices_tr], X_test[shuffledIndices_te], Y_train[shuffledIndices_tr], Y_test[shuffledIndices_te]
