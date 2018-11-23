# -*- coding: utf-8 -*-
'''Charles' regression code'''

import pandas as pd
import numpy as np

def Data_prep_replication(file):
    '''
    preparation of the data set in order to reproduce the regression form the paper

    data = Data_prep_replication("../Data/result_full_factorial_pgm.csv")


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
