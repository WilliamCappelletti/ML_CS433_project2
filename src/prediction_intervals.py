'''
Functions needed to compute the predction intervals using bootstraping
'''

import numpy as np
import pandas as pd


def bootstrap_resample(X, n=None):
    ''' Bootstrap resample an array_like
    Parameters
    ----------
    X : array_like
      data to resample
    n : int, optional
      length of resampled array, equal to len(X) if n==None

    Results
    -------
    returns X_resamples
    '''
    if isinstance(X, pd.Series):
        X = X.copy()
        X.index = range(len(X.index))
    if n == None:
        n = len(X)

    resample_i = np.floor(np.random.rand(n)*len(X)).astype(int)
    X_resample = np.array(X[resample_i])
    return X_resample


def prediction_interval(x,model, residuals , data, r= 100):
    ''' returns the prediction interval
    Parameters
    ----------
    x : data_like.
        used to get a prediction (e.g. model.predict(x))
    model : model from which we want to evaluate the prediction intervals
    residuals : array_like
        residuals from the training of the model : y - y_hat
    data:
        data on which model was trained
    r = bootstrap parameter

    Results
    -------
    returns [c_lowe,c_upper]
    '''

    #get the y_hat from the model
    y_hat = model.predict(x)

    Resi = []

    for r in range(R):

        #bootstrap from the residuals
        residuals_bootstraped = bootstrap_sample(residuals)

        #generate a new data_set to train the model on
        y_tilda = np.repeat(y_hat, len(residuals)) + residuals_bootstraped

        #train model on new data_set
        model_tilda = model_like.train(data,y_tilda)

        residuals_new = model_tilda.residuals

        Resi.append(np.random.choice(residuals_new))


    percentiles = np.percentile(Resi,[2.5,97.5])

    return percentiles

def prediction_intervals_array(x,model, residuals , data, r= 100):
    ''' returns the prediction interval for an array x
        useful to do plots
    Parameters
    ----------
    x : array of points.
        used to get a prediction (e.g. model.predict(x))
    model : model from which we want to evaluate the prediction intervals
    residuals : array_like
        residuals from the training of the model : y - y_hat
    data:
        data on which model was trained
    r = bootstrap parameter

    Results
    -------
    returns [c_lowe,c_upper] for each x_i
    '''

    results = []

    for point in x:

        results.append(prediction_interval(point,model, residuals , data, r))

    return results
