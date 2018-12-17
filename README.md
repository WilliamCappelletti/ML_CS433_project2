# ML_project_2

Second project for the course of machine learning CS433 in collaboration with the Laboratory of Computational Systems Biotechnology (SB/SV/STI) LCSB, based on the article "Particle-based simulation reveals macromolecular crowding effects on the Michaelis-Menten mechanism", by Weilandt, Daniel R and Hatzimanikatis, Vassily.

The purpose of this project is 

## Libraries used
We used the following libraries for this project:


 Computational:

    numpy (as np)
    scikit-leran (sklearn)
    Statsmodel
    Xgboost

Graphical:

    seaborn (as sns)
    matplotlib (as plt)


## Prerequisites

The folder structure has to be the following:

    .
    ├── Data                    # Data files, in .csv
        └── result_full_factorial_pgm.csv
    ├── src                     # Source files
        └── reproduction.py
    ├── results
    └── README.md


## Implementations details
### run_reproduction.py

Import function from `reproduction.py`

This script reproduce the model described in the original article, which is a weighted linear model (see the report for more details on the weigthing process).

It outputs a .csv file containing the estimate of the coefficient and store it in `../results/reproduction.csv`

## Links

overleaf report: https://www.overleaf.com/9168994853jnrbvbtpzrsf


### Notes on `cross_validation` and `multi_cross_validation`

These are the two main functions inplemented in order to choose our model, and in particular to get an estimation of the prediction error.

* `cross_validation(y, tx, k_fold, method, *args_method[, k_indices, seed])` compute the k-fold cross validation for the estimation of `y` using a the method-function stored (as pointer) in the argument `method`. The arguments necessary for the `method` are to be passed freely after method. It returns `predictor, w, loss_tr, loss_te`, which are, in order, the predicting function, the mean of the trained weights, the mean of the train error and the estimate test error.

* `multi_cross_validation(y, x, k_fold[, transformations=[[id, []]], methods=[[least_squares, []]], seed=1, only_best=True])` Perform automatically the cross validation on all the combinations of transformations in the `transformations` list (their parameters have to be passed as a list coupled with the transformation) and methods with changing parameters in the `methods` list (the coupled list have in this case to be a list of the tuples of parameters combinations to test.) It then plots the estimated losses (both on train and test) and outputs `predictor, weight, losses_tr, losses_te, transformations_list, methods_list`. If `only_best=True`, those are the variables corresponding to the lowest test-error estimate, otherwise they contain the variables computed at each step. An implementation example can be found in the documentation.


## References:

Particle-based simulation reveals macromolecular crowding effects on the Michaelis-Menten mechanism                                    
Daniel R Weilandt, Vassily Hatzimanikatis                                                                                           
doi: https://doi.org/10.1101/429316

Seabold, Skipper, and Josef Perktold. “Statsmodels: Econometric and statistical modeling with python.” Proceedings of the 9th Python in Science Conference. 2010.

Hastie, T., Tibshirani, R.,, Friedman, J. (2001). _The Elements of Statistical Learning_. New York, NY, USA: Springer New York Inc..



## Authors

* *William Cappelletti*
* *Charles Dufour*
* *Marie Sadler*
