# -*- coding: utf-8 -*-
'''William's regression code'''

import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score


def cast_categories(frame, column, cat_name = 'Group'):
    '''Returns a new data frame in which the specified column has been changed by new colums representing the appertenance to one specified category'''

    new_frame = frame.copy()
    categories = new_frame[column].unique()
    splitted_cols = {'{cat_name}_{x}'.format(cat_name=cat_name, x=x): [ 1 if inside else 0 for inside in new_frame[column]==x] for x in categories}
    splitted_cols = pd.DataFrame(splitted_cols, index = new_frame.index)
    new_frame = pd.concat([new_frame.drop(columns=[column]), splitted_cols], axis=1)
    return new_frame

def data_initialization():
    '''Read and split data.

    return y1, y2, y3, y4, X'''
    # define data path
    data_folder = '../Data/'

    # read the data
    data = pd.read_csv(data_folder +'result_full_factorial_pgm.zip', index_col = 0)
    data.drop('Unnamed: 0.1', axis=1, inplace=True)

    y1, y2, y3, y4 = np.log(data['k1_bwd_effective']), np.log(data['k1_fwd_effective']), np.log(data['k2_bwd_effective']), np.log(data['k2_fwd_effective'])
    # realization = data['realization']

    X = data.drop(columns=['k1_bwd_effective','k1_fwd_effective','k2_bwd_effective','k2_fwd_effective',
                      'k1_bwd_relative','k1_fwd_relative','k2_bwd_relative','k2_fwd_relative', 'realization'])

    X = cast_categories(X, 'sigma_mass', 'sigma_mass')

    X[['enzyme_complex_concentration', 'enzyme_concentration', 'product_concentration',
       'substrate_concentration']] = X[['enzyme_complex_concentration', 'enzyme_concentration', 'product_concentration',
       'substrate_concentration']].apply(lambda x : np.log(x))

    return y1, y2, y3, y4, X

def polynomial_data(X, deg=2, interaction_only=False):
    X1 = X[['enzyme_complex_concentration', 'enzyme_concentration', 'mu_mass', 'product_concentration',
       'substrate_concentration', 'volume_fraction']].values
    X2 = X[['sigma_mass_0.0', 'sigma_mass_0.825']].values
    X3 = np.concatenate((X2[:,0].reshape(-1,1) * X1, X2[:,1].reshape(-1,1) * X1), axis=1)
    poly = PolynomialFeatures(deg, interaction_only=interaction_only, include_bias=False)
    X1 = poly.fit_transform(X1)
    return np.concatenate((X1, X3, X2), axis = 1)

def module_test():
    '''Function to test module inclusion'''
    return 0
