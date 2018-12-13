'''
 script to reproduce the plots from the report
'''

import pandas as pd
import os
import random
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns


from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score
import regressions

from matplotlib import rcParams

sns.set()
rcParams['axes.titlepad'] = 20



from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import numpy as np
import xgboost as xgb




#-------------------------------------------------------------------------------
# Load the data
#-------------------------------------------------------------------------------




#-------------------------------------------------------------------------------
# constants definition
#-------------------------------------------------------------------------------
# heuristic for the xgboost model, taken from the median of the training data

P0 = np.median(X_panda.product_concentration.values)
S0 = np.median(X_panda.substrate_concentration.values)
ES0 = np.median(X_panda.enzyme_complex_concentration.values)
mu0 = np.median(X_panda.mu_mass.values)
v0 = 0.3

#-------------------------------------------------------------------------------
# Train the models and save the plots
#-------------------------------------------------------------------------------
