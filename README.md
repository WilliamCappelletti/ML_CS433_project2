# ML_project_2

Second project for the course of machine learning CS433 in collaboration with the Laboratory of Computational Systems Biotechnology (SB/SV/STI) LCSB, based on the article "Particle-based simulation reveals macromolecular crowding effects on the Michaelis-Menten mechanism", by Weilandt, Daniel R and Hatzimanikatis, Vassily.

The purpose of this project is to find a model to unravel the effects of crowding on the kinetic of biochemical reactions. The lab provided us with simulated data: the software GEneralized Elementary Kinetics (GEEK), predicts the enzymatic rate constants for varying volume exclusion conditions (i.e. the space occupied by inert molecules representative for a crowded environment),mass and mass distribution of the inert  molecules,  and  reactive  species  concentrations.  Starting from this simulation data, our study aims to develop a model predicting the rate constants in a black box machine learning approach.

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


To install some of the libraries mentionned before , please use the following command (linux):

    pip3 install xgboost
    pip install -U statsmodels

The folder structure has to be the following:

    .
    ├── Data                    # Data files, in .csv
        └── result_full_factorial_pgm.csv
    ├── src                     # Source files
        └── reproduction.py
    ├── results
         └── plots_evolution
    └── README.md


## Implementations details
### run_reproduction.py

Import function from `reproduction.py`

This script reproduce the model described in the original article, which is a weighted linear model (see the report for more details on the weigthing process).

It outputs a .csv file containing the estimate of the coefficient and store it in `../results/reproduction.csv`

### marie.py 

explains what it does

### run_plots.py 

Import function from `plots.py`

This script plots the prediction made by our final mdel xgboost. More precisely, having fixed all the input of our model except two concentrations, we plot the value of the kinetic constant in color, while the axes represent the evolution of the tow concentrations we chose before.

The fixed values are usually fixed as being the median of what we had in our training data (`result_full_factorial_pgm.csv`).

The script produces the plots for all pair of concentrations, for each volume fraction and for each output.

It saves the results in `../results/plots_evolution`, then classes the plots by the value of the volume fraction considered.

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
