# -*- coding: utf-8 -*-
'''Charles' regression code'''

import pandas as pd
import numpy as np
import statsmodels.api as sm

def Data_prep_replication(file):
    '''
    preparation of the data set in order to reproduce the regression form the paper

    data = Data_prep_replication("../Data/result_full_factorial_pgm.csv")

    returns the modified data_set taking into account the conflicts of notations between the paper
    and the data_set
    '''

    data = pd.read_csv(file)

    #drop the first two columns
    data.drop('Unnamed: 0.1', axis=1, inplace=True)
    data.drop("Unnamed: 0",axis = 1, inplace = True)

    # create transformed dataframe
    data_trans = data.copy()

    #log of the ratio
    data_trans['log_k1_bwd'] = np.log(data_trans['k1_bwd_relative'])
    data_trans['log_k1_fwd'] = np.log(data_trans['k1_fwd_relative'])
    data_trans['log_k2_bwd'] = np.log(data_trans['k2_bwd_relative'])
    data_trans['log_k2_fwd'] = np.log(data_trans['k2_fwd_relative'])


    #computation of initial concentration
    E_tot = 64e-6

    P0 = 49e-6
    S0 = 49e-6

    E0 = 0.5*E_tot
    ES0 = 0.5*E_tot



    #computation of the log of the ration of the concentration
    data_trans['E'] = np.log(data_trans['enzyme_concentration']/E0)
    data_trans['ES'] = np.log(data_trans['enzyme_complex_concentration']/ES0)
    data_trans['P'] = np.log(data_trans['product_concentration']/P0)
    data_trans['S'] = np.log(data_trans['substrate_concentration']/S0)



    #dropping the covariates we don't need anymore
    data_trans.drop(['k1_bwd_effective','k1_bwd_relative', 'k1_fwd_effective',
                    'k1_fwd_relative', 'k2_bwd_effective', 'k2_bwd_relative',
                    'k2_fwd_effective', 'k2_fwd_relative', 'enzyme_concentration',
                    'enzyme_complex_concentration', 'product_concentration',
                    'substrate_concentration','realization'], axis=1, inplace=True)

    return data_trans


def weights_obtain(data):
    '''
    weights = weights_obtain(data_test)
    data should be of the form that Data_prep_replication() returns


    return the weights necessary to replicate the weighted linear model of the paper for
    each regression
    '''
    weights = pd.DataFrame()

    #build the regression matrix
    concentrations = ['E','ES','P','S']
    X = data[concentrations]
    X = sm.add_constant(X)

    targets = ['log_k1_bwd','log_k1_fwd','log_k2_bwd','log_k2_fwd']

    for name in targets:

        y = data[name]
        model = sm.OLS(y,X).fit()

        #build a data frame containing the residuals and the fitted values
        data_residuals = pd.DataFrame(dict(fittedvalues=model.fittedvalues,
                                   residuals=model.resid))

        # the observations are grouped depending on the fitted values they have approximately
        data_residuals['groups'] = (data_residuals.fittedvalues/
                                    max(abs(data_residuals.fittedvalues))).round(1)

        #we compute the standard deviation in the different groups
        sd_groups = data_residuals.groupby('groups')['residuals'].std()

        data_residuals['standard_deviation'] = np.ones(len(data_residuals.fittedvalues))

        #assign to each data_point the standard deviation of the group it belongs to
        for group in sd_groups.index.values:
            data_residuals['standard_deviation'][data_residuals['groups']==group] = sd_groups.ix[group]

        #define the weight as the inverse of the std and store it
        weight = 1.0/data_residuals['standard_deviation']
        weights[name] = weight

    return weights
