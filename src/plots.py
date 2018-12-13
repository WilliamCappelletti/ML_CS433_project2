'''
Functions needed to obtain the plots from the report
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
# constants for the plots
#-------------------------------------------------------------------------------

limits_d ={ 0.0: {'y1': 2.27,'y2': 10.55,'y3':11.46,'y4':3.06},
            0.1: {'y1': 1.94,'y2': 10.6, 'y3':11.12,'y4':2.72},
            0.2: {'y1': 1.42,'y2': 9.25, 'y3':11.82,'y4':2.21},
            0.3: {'y1': 0.60,'y2': 10.98,'y3':11.8, 'y4':1.39},
            0.4: {'y1':-0.57,'y2': 11.56,'y3':11.59,'y4':0.21},
            0.5: {'y1':-1.54,'y2': 12.26,'y3':12.51,'y4':-0.75}}

limits_u ={ 0.0: {'y1': 2.3, 'y2': 12.69,'y3':13.7,'y4':3.09},
            0.1: {'y1': 2.21,'y2': 13.13,'y3':13.79,'y4':2.99},
            0.2: {'y1': 2.06,'y2': 13.42,'y3':14.22,'y4':2.85},
            0.3: {'y1': 1.81,'y2': 13.56,'y3':14.29,'y4': 2.6},
            0.4: {'y1': 1.38,'y2': 13.75,'y3':14.53,'y4': 2.16},
            0.5: {'y1': 0.15,'y2': 13.86,'y3':14.59,'y4': 0.94}}


def compute_mesh_values(x, y, base, covariates, model):
    '''

    exampple of input:
        x = arange(-13,-6,0.01),y = arange(-13,-6,0.01), base = np.array([ES0,mu0,P0,S0,v0]),covariates = ['P','S'], model = xgbReg
    '''
    # the function that I'm going to plot
    def z_func(x,y):

        ES_ref = np.repeat(base[0],len(x))
        P_ref = np.repeat(base[2],len(x))
        mu_ref = np.repeat(base[1],len(x))
        v_ref = np.repeat(base[4],len(x))
        S_ref = np.repeat(base[3],len(x))

#---------------------------------------------------------------------------------------------
# choose the plot with the right axes

        if covariates == ['P','S']:
            predict_on = np.array([ES_ref,mu_ref,x,y,v_ref])
        elif covariates == ['S','P']:
            predict_on = np.array([ES_ref,mu_ref,y,x,v_ref])

        elif covariates == ['ES','P']:
            predict_on = np.array([x,mu_ref,y,S_ref,v_ref])
        elif covariates == ['P','ES'] :
            predict_on = np.array([y,mu_ref,x,S_ref,v_ref])


        elif covariates == ['ES','S'] :
            predict_on = np.array([x,mu_ref,P_ref,y,v_ref])
        elif covariates == ['S','ES']:
            predict_on = np.array([y,mu_ref,P_ref,x,v_ref])
        else:
            raise Exception('combinaison not implemented')

#----------------------------------------------------------------------------------------------

        dtest = xgb.DMatrix(np.matrix.transpose(predict_on))
        result = np.ndarray.round(model.predict(dtest),3)
        return result


    def Z_FUNC(X,Y):
        x = np.array([(x, y) for x in X for y in Y])
        result = z_func(x[:,0],x[:,1]).reshape(len(X),len(Y))
        return result

    X,Y = meshgrid(x, y) # grid of point

    Z = Z_FUNC(x, y) # evaluation of the function on the grid

    return np.transpose(Z)


def plot_two_covariates(x, y, base, covariates, model,name):
    '''

    example of input:
        x = arange(-13,-6,0.01), y = arange(-13,-6,0.01), base = np.array([ES0,mu0,P0,S0,v0]), covariates = ['P','S'], model = xgbReg
    '''

    thisdict =	{"y1": "log(k1_bwd)", "y2": "log(k1_fwd)", "y3": "log(k2_bwd)", "y4": "log(k2_fwd)"}

    Z = compute_mesh_values(x,y,base,covariates, model)

    vol = base[-1]

    _min = limits_d[vol][name]
    _max = limits_u[vol][name]
    print("borne for the plots: ",_min,_max)

    im = imshow(Z,cmap=cm.RdBu)#,vmin = _min, vmax = _max) # drawing the function

    #colorbar(im)
    ax = plt.gca()

    cbar = colorbar(im)
    cbar.ax.set

    ax.set_xticks(np.arange(0,len(x),len(x)/3))
    ax.set_yticks(np.arange(0, len(x), len(x)/3))
    ax.set_xticklabels([round(x[0],2),round(x[int(len(x)/3)],2),round(x[int(2*len(x)/3)],2)])
    ax.set_yticklabels([round(y[0],2),round(y[int(len(x)/3)],2),round(y[int(2*len(x)/3)],2)])
    plt.xlabel(covariates[0])
    plt.ylabel(covariates[1])

    plt.title("evolution of "+ thisdict[name]+" wrt to [" + covariates[0] + '] and [' + covariates[1] +']' )
    #plt.savefig('../results/plots_evolution/volume'+str(base[-1])+'/'+name +covariates[0]+'_'+ covariates[1]+'.pdf', bbox_inches='tight')
    plt.show()


def plot_evolution(model_parameters, X,y,name, base):

    dtrain1 = xgb.DMatrix(X, label=y)
    xgbReg = xgb.train(params=model_parameters, dtrain=dtrain1)


    volumes = np.array([0. , 0.1, 0.2, 0.3, 0.4, 0.5])
    concentrations = ['ES','P','S']

    index = {'ES':0,'P':2,'S':3}

    for v in volumes :
        print(" ")
        print("VOLUME ",v)
        print(" ")
        base[-1] = v
        done = []
        for i,c1 in enumerate(concentrations):
            for c2 in np.delete(concentrations,i):
                candidats = [c1,c2]
                if candidats not in done:
                    done.append(candidats)
                    done.append([c2,c1])

                    i1 = index[c1]
                    i2 = index[c2]

                    x_ax = X[:,i1]
                    y_ax = X[:,i2]

                    x1 = np.arange(np.min(x_ax)-2,np.max(x_ax)+2,0.01)
                    x2 = np.arange(np.min(y_ax)-2,np.max(y_ax)+2,0.01)


                    plot_two_covariates(x1,x2,base = base, covariates = [c1, c2], model = xgbReg, name = str(name))
                else :
                    print("combinaison already done: ",c1, " ",c2)
