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
import plots




#-------------------------------------------------------------------------------
# Load the data
#-------------------------------------------------------------------------------

y1, y2, y3, y4, X, realization = regressions.data_initialization(realiz=True)
X = X.drop(columns=['sigma_mass_0.0', 'sigma_mass_0.825', 'enzyme_concentration'])
X_panda = X.copy()
X = X.values


#-------------------------------------------------------------------------------
# constants definition
#-------------------------------------------------------------------------------
# heuristic for the xgboost model, taken from the median of the training data

P0 = np.median(X_panda.product_concentration.values)
S0 = np.median(X_panda.substrate_concentration.values)
ES0 = np.median(X_panda.enzyme_complex_concentration.values)
mu0 = np.median(X_panda.mu_mass.values)
v0 = 0.3

#parameters for the model xgboost obtained via cross validation
param = {'max_depth': 8, 'eta':  0.6000000000000001, 'silent': 1, 'subsample': 1.0}
# param['nthread'] = 4
param['reg_alpha'] = 0.5
# param['reg_lamda'] = 0.5
param['tree_method'] = 'auto'

#-------------------------------------------------------------------------------
# Train the models and save the plots
#-------------------------------------------------------------------------------
for y,name in  [[y1,'y1'],[y2,'y2'],[y3,'y3'],[y4,'y4']]:
    print("training and plotting for response ",name)
    plots.plot_evolution(model_parameters= param,X = X,y = y,name = name,base = np.array([ES0,mu0,P0,S0,v0]))
