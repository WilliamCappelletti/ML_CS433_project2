# -*- coding: utf-8 -*-
'''code needed to reproduce the weighted linear regression from the original paper'''

import pandas as pd
import numpy as np
import statsmodels.api as sm

#-------------------------------------------------------------------------------
# utilities to do the reproduciton model
#-------------------------------------------------------------------------------
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
    each regression: each observations is first cast into groups depending on its residual value, then
    it is weighted as 1/ variance of the group it has been cast into.
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


def regression_results(data):
    '''
    Function to produce the regression results from the table in the supplementary material
    Input: prepared dataframe (output from: Data_prep_replication(FILE))
    '''

    # initialize all the combinations
    Mass_median_sigma = [[31.9, 0.825], [12.1, 0], [21.1, 0], [36.8, 0]]
    volume_fraction = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    rate_constants = ['log_k1_bwd','log_k1_fwd','log_k2_bwd','log_k2_fwd']
    rate_constant_name = ['k1_bwd','k1_fwd','k2_bwd','k2_fwd']

    # initialze result list
    regression_results = []

    # generate the results
    for m_index, m in enumerate(Mass_median_sigma):
        for v in volume_fraction:

                # condition to execute v = 0 only once
                if (v == 0) & (m_index != 0):
                    continue
                # select the data of the defined conditions
                data_reg = data[(data['volume_fraction']==v) & (data['mu_mass'] == m[0]) & (data['sigma_mass']==m[1])]
                weights = weights_obtain(data_reg)

                # build the model matrix
                X = data_reg[['E','ES','P','S']]
                X = sm.add_constant(X)

                # do the regression for each rate constant
                for name, k in enumerate(rate_constants):

                    # fit the model
                    model_k = sm.WLS(data_reg[k],X, weights=weights[k]).fit()

                    # store regression results
                    results_reg = 20*[0]
                    results_reg[0:20:4] = model_k.params
                    results_reg[1:20:4] = model_k.conf_int(alpha=0.05)[0]
                    results_reg[2:20:4] = model_k.conf_int(alpha=0.05)[1]
                    results_reg[3:20:4] = model_k.pvalues

                    # create result vector with conditions, rate constant name & regression result
                    result = m + [v, rate_constant_name[name]]
                    result = result+results_reg
                    # append the result to the dataframe
                    regression_results.append(result)

    # convert the result to pandas dataframe
    regression_results_df = pd.DataFrame(regression_results)

    # add a header
    header = [np.array(['Median of the Massdistribution in kDa', 'Sigma parameter of the Massdistribution', 'Volume fraction','Rate constant',
                    'beta','beta','beta', 'beta', 'alpha E', 'alpha E', 'alpha E', 'alpha E',
                   'alpha ES', 'alpha ES', 'alpha ES', 'alpha ES', 'alpha P', 'alpha P',
                    'alpha P', 'alpha P', 'alpha S', 'alpha S', 'alpha S', 'alpha S']),
    np.array(['','','','','Estimate','0.025','0.975', 'p-value', 'Estimate','0.025','0.975',
              'p-value', 'Estimate','0.025','0.975', 'p-value', 'Estimate','0.025','0.975', 'p-value',
             'Estimate','0.025','0.975', 'p-value'])]

    regression_results_df.columns=header

    return regression_results_df


#-------------------------------------------------------------------------------
# goodness of fit measure
#-------------------------------------------------------------------------------
def obtain_measure_goodness_of_fit_reproduction():
    '''
    obtain the measures of goodness of fit of the weigthed linear regression that are in the report

    returns R2c, MSE:
        R2c contains the R2 adjusted of each model. This is defined here as 1 - ssr/centered_tss if the constant is included in the model and 1 - ssr/uncentered_tss if the constant is omitted.
        MSE contains the total mean squared error. Defined as the uncentered total sum of squares divided by n the number of observations.

    for each volume fraction tested ([0.1,0.2,0.3,0.4,0.5]]) it will return a vector with the data corresponding to the model in the following order:
        ['log_k1_bwd','log_k1_bwd','log_k2_bwd','log_k2_fwd']
    '''

    file = "../Data/result_full_factorial_pgm.csv"

    data = Data_prep_replication(file)


    #do the regression for some fixed value of mu and sigma

    mu = 31.9
    sigma = 0.825

    volumes = [0.1,0.2,0.3,0.4,0.5]

    R2c = []
    MSE = []

    for volume in volumes:

        #create empty container for stocking the measures for this volume fraction
        r2c = []
        mse = []

        #select the subset of the data we are interested in
        data_test = data[(data['volume_fraction']==volume) & (data['mu_mass']==mu) & (data['sigma_mass']==sigma)]

        #build the regression matrix
        X = data_test[['E','ES','P','S']]
        X = sm.add_constant(X)

        #compute the weights for the wlr on these data
        Weights = weights_obtain(data_test)

        #fit the weighted linear model for each response
        model_y1 = sm.WLS(data_test['log_k1_bwd'],X, weights=Weights.log_k1_bwd).fit()
        model_y2 = sm.WLS(data_test['log_k1_fwd'],X, weights=Weights.log_k1_fwd).fit()
        model_y3 = sm.WLS(data_test['log_k2_bwd'],X, weights=Weights.log_k2_bwd).fit()
        model_y4 = sm.WLS(data_test['log_k2_fwd'],X, weights=Weights.log_k2_fwd).fit()

        #create an iterable on the models
        models = [model_y1,model_y2,model_y3,model_y4]

        for m in models:
            #collect the r squared
            r2c.append(m.rsquared)
            #collect the mse of the model on its train set
            mse.append(m.mse_total)

        R2c.append(r2c)
        MSE.append(mse)

    R = pd.DataFrame(R2c)
    M = pd.DataFrame(MSE)

    #adding the volume fraction in the dataframe
    R['volume fraction']= [0.1,0.2,0.3,0.4,0.5]
    M['volume fraction']= [0.1,0.2,0.3,0.4,0.5]

    #naming the columns
    header = [np.array(['log_k1_bwd','log_k1_fwd','log_k2_bwd','log_k2_fwd','volume fraction'])]
    R.columns=header
    M.columns = header

    #rearranging the order of the columns
    R = R[['volume fraction','log_k1_bwd','log_k1_fwd','log_k2_bwd','log_k2_fwd']]
    M = M[['volume fraction','log_k1_bwd','log_k1_fwd','log_k2_bwd','log_k2_fwd']]


    return R,M
