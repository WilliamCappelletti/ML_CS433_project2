# -*- coding: utf-8 -*-
'''Reproduction of the figure 2 of the supplementary material (rate constants vs. concentrations)'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import reproduction

def data_prep_fig2_subplot(file):
    '''Selects and prepares the data needed to plot 1 subfigure of the figure 2 of the supplementary material.
    Example: data_k_conc = data_prep_fig2_supp("../Data/result_full_factorial_pgm.csv")'''

    data = pd.read_csv(file)

    #drop the first two columns
    data.drop('Unnamed: 0.1', axis=1, inplace=True)
    data.drop("Unnamed: 0",axis = 1, inplace = True)

    # create transformed dataframe
    data_trans = data.copy()

    # rename concentration columns
    data_trans = data_trans.rename(columns={'enzyme_concentration': 'E', 'enzyme_complex_concentration': 'ES',
                                            'product_concentration': 'P', 'substrate_concentration': 'S'})


    # select data with volume fraction 0%, 30% and 50%
    data_k_conc = data_trans[data_trans.volume_fraction.isin([0,0.3,0.5])].copy()
    data_k_conc[['E', 'ES', 'P', 'S']] = data_k_conc[['E', 'ES', 'P', 'S']]*10e6
    return data_k_conc


def plot_fig2_subplot(data_k_conc, k, k_label, conc, conc_label, save = False):
    '''Plots 1 subplot of the figure 2 of the supplementary data.
    data_k_conc should be the output of the function 'data_prep_fig2_subplot'
    k can take one of the following variable names: 'k1_bwd_relative', 'k1_fwd_relative', 'k2_bwd_relative', 'k1_bwd_relative'
    conc can take: 'E', 'ES', 'S', 'P'
    k_label and conc_label are the desired axes labels'''

    # prepare figure with colour condition on the volume fraction
    fg = sns.FacetGrid(data=data_k_conc, hue='volume_fraction', size = 4, aspect = 1.4,
                 palette= {0: 'b', 0.3: "#b54334", 0.5: "g"})
    ax = fg.axes[0][0]
    ax.set(xscale="log", yscale="log")

    # plot the rate constant vs. concentration
    fg.map(plt.scatter, conc, k).add_legend(title = 'Volume fraction')

    # title and axis labels
    ax.set_xlabel(conc_label, fontsize = 12)
    ax.set_ylabel(k_label, fontsize = 12)
    if save:
        fg.savefig("../results/Fig2_report.png", dpi= 300)
    return

def data_prep_fig2_plot(file):
    '''Selects and prepares the data needed to plot the entire figure 2 of the supplementary material.
    Example: data_k_conc = data_prep_fig2_supp("../Data/result_full_factorial_pgm.csv")'''

    data = pd.read_csv(file)

    #drop the first two columns
    data.drop('Unnamed: 0.1', axis=1, inplace=True)
    data.drop("Unnamed: 0",axis = 1, inplace = True)

    # create transformed dataframe
    data_trans = data.copy()

    # rename concentration columns
    data_trans = data_trans.rename(columns={'enzyme_concentration': '[E] [µM]', 'enzyme_complex_concentration': '[ES] [µM]',
                                            'product_concentration': '[P] [µM]', 'substrate_concentration': '[S] [µM]',
                                           'k2_bwd_relative': '$k_{2,b,eff}/k_{2,b,0}$', 'k1_bwd_relative': '$k_{1,b,eff}/k_{1,b,0}$',
                                           'k2_fwd_relative': '$k_{2,f,eff}/k_{2,f,0}$', 'k1_fwd_relative': '$k_{1,f,eff}/k_{1,f,0}$'})
    data_trans = data_trans.drop(['mu_mass', 'realization', 'sigma_mass', 'k2_fwd_effective', 'k1_fwd_effective', 'k1_bwd_effective', 'k2_bwd_effective'], axis=1)
    # select data with volume fraction 0%, 30% and 50%
    data_k_conc = data_trans[data_trans.volume_fraction.isin([0,0.3,0.5])].copy()
    data_k_conc[['[E] [µM]', '[ES] [µM]', '[P] [µM]', '[S] [µM]']] = data_k_conc[['[E] [µM]', '[ES] [µM]', '[P] [µM]', '[S] [µM]']]*10e6
    return data_k_conc

def plot_fig2_plot(file, save = False):
    '''Plots the entire figure 2 of the supp. material: all the 4 rate constants with respect to the 4 concentrationsself.
    Example: plot_fig2_plot('../Data/result_full_factorial_pgm.csv')'''

    data = data_prep_fig2_plot(file)
    # change the data to levels
    data_melt = pd.melt(data, id_vars=['$k_{2,b,eff}/k_{2,b,0}$', '$k_{1,b,eff}/k_{1,b,0}$',
                                               '$k_{2,f,eff}/k_{2,f,0}$', '$k_{1,f,eff}/k_{1,f,0}$', 'volume_fraction'], value_vars=['[E] [µM]', '[ES] [µM]', '[P] [µM]', '[S] [µM]'],
                         var_name='concentration', value_name='concentration_value')
    data_melt2 = pd.melt(data_melt, id_vars = ['concentration', 'concentration_value', 'volume_fraction'], value_vars = ['$k_{2,b,eff}/k_{2,b,0}$', '$k_{1,b,eff}/k_{1,b,0}$',
                                               '$k_{2,f,eff}/k_{2,f,0}$', '$k_{1,f,eff}/k_{1,f,0}$'], var_name='k', value_name = 'k_value')

    # prepare the figure
    fg = sns.FacetGrid(data_melt2, col='concentration', row = 'k', hue='volume_fraction',
                       sharex=False, sharey=False, size=4, aspect = 0.9,
                     palette= {0: 'b', 0.3: "#b54334", 0.5: "g"})

    # change to logarithmic axes
    for ax in fg.axes.flat:
        ax.set(xscale="log", yscale="log")
    fg = (fg.map(plt.scatter, "concentration_value", "k_value", edgecolor="w")).add_legend(title = 'Volume fraction')\
            .set_titles('')

    # x axes titles
    fg.axes[3,0].set_xlabel('[E] [µM]')
    fg.axes[3,1].set_xlabel('[ES] [µM]')
    fg.axes[3,2].set_xlabel('[P] [µM]')
    fg.axes[3,3].set_xlabel('[S] [µM]')

    # y axes titles
    fg.axes[0,0].set_ylabel('$k_{2,b,eff}/k_{2,b,0}$')
    fg.axes[1,0].set_ylabel('$k_{1,b,eff}/k_{1,b,0}$')
    fg.axes[2,0].set_ylabel('$k_{2,f,eff}/k_{2,f,0}$')
    fg.axes[3,0].set_ylabel('$k_{1,f,eff}/k_{1,f,0}$')

    #plt.show()
    # save the figure
    if save:
        fg.savefig("../results/Fig2_supp.png", dpi= 300)
    return
