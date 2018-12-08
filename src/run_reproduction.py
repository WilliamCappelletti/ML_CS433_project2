'''
Reproduction of the weighted linear regression done in the original paper
'''


import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib as plt
import pandas as pd


from reproduction import Data_prep_replication,regression_results,weights_obtain

# # -------------------------------------------------------------------------------
# reproduction of the regression
# -------------------------------------------------------------------------------

#load file
file = "../Data/result_full_factorial_pgm.csv"

#prepare the data_set
data = Data_prep_replication(file)

#computation of the coefficients
results = regression_results(data)


results.to_csv('../results/reproduction.csv')
