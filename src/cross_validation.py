# -*- coding: utf-8 -*-
'''Cross validation function for hyperparameters tuning'''

import numpy as np
import matplotlib.pyplot as plt


#**************************************************
# CROSS VALIDATION
#--------------------------------------------------

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = len(y)
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation_visualization(steps, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.plot(steps, mse_tr, marker=".", color='b', label='train error')
    plt.plot(steps, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("CV step")
    plt.ylabel("loss")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    #plt.savefig("cross_validation")

def single_validation(X, y, k_indices, k, method, **method_params):
    """
    Train given model on all subset of k_indices but k-th and compute prediction error over k-th subset of k_indices.
    Returns predictor, w, single_loss_tr, single_loss_te
    """

    # get k'th subgroup in test, others in train
    x_te = X[k_indices[k]]
    y_te = y[k_indices[k]]
    k_complement = np.ravel(np.vstack([k_indices[:k],k_indices[k+1:]]))
    x_tr = X[k_complement]
    y_tr = y[k_complement]

    # regression using the method given
    method.fit(x_tr, y_tr, **method_params)

    # calculate the loss for train and test data
    single_loss_tr = method.score(x_tr, y_tr)
    single_loss_te = method.score(x_te, y_te)
    return single_loss_tr, single_loss_te

def cross_validation(X, y, method, k_fold, k_indices=None, seed=1, **method_params):
    '''
    return an estimate of the expected predicted error outside of the train set for the model, using k fold cross validation
    *args_model are the parameter needed for the model to train (for example lambda for ridge,..,)
    estimate = CV(y, tx, k_fold[, k_indices, loss_f, err_f,], model, *args_model)
    prediction = model(x_test,y_train,x_train,*args_model): model is a function that return the prediction classification for a specific modelself.

    Return predictor, w, loss_tr, loss_te
    '''

    if k_indices is None:
        k_indices = build_k_indices(y, k_fold, seed = seed)

    single_loss_tr = np.zeros(k_fold)
    single_loss_te = np.zeros(k_fold)

    for k in range(k_fold):
        single_loss_tr[k], single_loss_te[k] = single_validation(X, y, k_indices, k, method, **method_params)
    loss_tr = np.mean(single_loss_tr)
    loss_te = np.mean(single_loss_te)

    return loss_tr, loss_te


def multi_cross_validation(X, y, methods, k_fold, seed=1, only_best=True):
    '''
        Run cross validation for whatever you can think of (combination of models, different transformations for the features... sorry it doesn't cure cancer yet @william ;) )

        Return predictors, ws, losses_tr, losses_te, t_list, m_list. (Only best value if only_best=True)

        example of use:

            methods = [[easyXGB, [{'reg_alpha': 0.7, 'max_depth': 10},
                                {'reg_alpha': 0.5, 'max_depth': 8}]]]

            loss_tr, loss_te, method = multi_cross_validation(X, y, methods=methods, k_fold, seed=2)

    '''
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    losses_tr = []
    losses_te = []
    m_list = []


    for method, parameters in methods:
        for m_params in parameters:
            print('Testing for method {name} with param.s {m_params}... Be patient! ;)'.format(name=method, m_params=m_params))
            loss_tr, loss_te = cross_validation(X, y, method, k_fold, k_indices=k_indices, **m_params)
            losses_tr.append(loss_tr)
            losses_te.append(loss_te)
            m_list.append([method, m_params])


    cross_validation_visualization(range(len(losses_tr)), losses_tr, losses_te)

    #Routine to gest just best hyper_parameter
    if only_best:
        best_i = np.argmin(losses_te)
        return losses_tr[best_i], losses_te[best_i], m_list[best_i]
    return losses_tr, losses_te, m_list
