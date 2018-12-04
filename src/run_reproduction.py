'''
Reproduction of the weighted linear regression done in the original paper
'''


import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib as plt
import pandas as pd

#-------------------------------------------------------------------------------
# functions definition
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


#-------------------------------------------------------------------------------
# reproduction of the regression
#-------------------------------------------------------------------------------

#load file
file = "../Data/result_full_factorial_pgm.csv"

#prepare the data_set
data = Data_prep_replication(file)

#computation of the coefficients
results = regression_results(data)


results.to_csv('../results/reproduction.csv')
