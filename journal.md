# up to 20.11.2018
what has been done:
1. tries to replicate, resulted in failed attempts


2. after discussion with the supervisor, we finally managed to figure out what was all the replication about: below is a résumé.

  - There are notation conflicts between the provided data and the paper. In the provided dataset kj/kj0 corresponds to kj_relative. So basically what we where doing: we divided kj/(kj/k0) = k0,
 which then is a constant.

  - Concentrations X_0: they are now corrected in the cell below. The concentrations in the dataset are in Molar (not micromolar). E_0 and ES_0 are 0.5 * E_tot (which he assumed was not written
 in the paper), and the reason for this number has to do with the saturation limit in the Michaelis Menten mechanism.

  - The parameters volume fraction, sigma_mass and mu_mass are all related to the inert crowding particles.

  - Volume fraction: how much space is crowded by inert particles. 0.5 is the limit -> beyond that we have a solid. In the volume fraction of 0 condition, there are no inert particles, so this condition
 is independent of sigma_mass and mu_mass.

  - Sigma_mass: This variable is related to the variation in the mass distribution of the inert particles. If it is 0, all the inert particles have the same size. We can treat this variable as categorical

  - Mu Mass: this is the median of the mass distribution. So if sigma_mass is 0, all the inert particles have the mass given by this parameter. This is a continuous variable. **He told me that simulations
 below 12.1 fail, but he will do further simulations above 36.8 next week. He expects, that if he increases the mass, at some point, there will be a limit where things change.**

  - Realization number: This is the index of the random seed. We can ignore this variable. However, if there is a correlation between the realization and any other variable, there is a big problem
 (everyone is welcome to check that).

  - And now the important part about the regression: Already **there is a mistake in the labelling of the heteroscedasticity figure**, the x-axis should by yi and not yi_hat.
 The y-axis is (y_hat-y)/y (He divides by y to have nice numbers between -1 and 1). Now, the procedure of how he does this conditional stuff: He fits a linear regression on the WHOLE MODEL
(alpha on the 4 concentrations and separately for each condition of mu_mass, volume_fraction, sigma_mass, and for each kj), and then he attributes the values of yi (not fitted y) into 10 bins.
 In each bin, he calculates the variance (sqrt(RSS/n-2), he only uses VAR from the package) and then 1/variance will be a weight_i attributed to yi. So in the end, every yi has in addition a wi,
 and then he fits a **weighted linear regression** (simply using statsmodel library).

  - **Yesterday he created an open Github with his regression script (it really exists, I saw the script)**:  https://github.com/EPFL-LCSB/geek

# 23.11.2018

- Charles has coded a function to modify the data in a proper form to do the replication of the paper

# 27.11.2018
- Charles coded a function to return the weights used in the linear regression in the paper, has to be checked: gives results very close to the results in the csv files in the supplement information, but not quite that

- Marie has coded a function to return the data needed to plot the supplementary figure 2 and another function which plots 1 subplot of this figure (given a k and a concentration). The resulting subplots has a larger variation in the data than the plot in the paper.
Question: Is the data already conditionally transformed in the plot supp. Fig. 2, and thus the variation is smaller?
Enhancement: Write code to generate the whole figure (16 subplots) for all the combinations of rate constants and concentrations.

# 28.11.2018

- _William_ coded function `cast_categories`, should be easy to use.

# 28.11.2018
- Marie created a function 'regression_results' to generate the dataframe with the regression results from the supplementary data. A demonstration is in Marie_test together with a codeline to ouput the corresponding csv file.
The resulting values are almost exactly the same than in the supplementary data.

# 29.11.2018

- _William_ tried RidgeCV on simple set and w/ interactions. Nice results (0.09525 MSE on test set).
- _William_ tested various regressions (will be added to .py in future). As for now best performance on `k1_bkw_effective` (test MSE *0.0129*, R\^2 0.98833) with RidgeCV polynomial deg 2 and all interactions. All concentrations are log transformed as well as the response.

# 02.12.2018

- _Charles_ coded the `run_reproduction.py` which reproduces the results of the paper and save them in `results/reproduction.csv`

# 3.12.2018

- _Marie_ has written the introduction of the report (up to the current knowledge of the project frame)
- _Marie_ created a function "plot_fig2_plot" to plot the whole figure 2 of supplementary materials (16 subplots). Resulting figure is saved in "../results/Fig2_supp.png"
