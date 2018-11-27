# -*- coding: utf-8 -*-
'''Marie's regression code'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def data_prep_fig2_supp(file):
    '''Selects and prepares the data needed to plot the figure 2 of the supplementary material.
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


def plot_fig2_supp(data_k_conc, k, k_label, conc, conc_label):
    '''Plots 1 subplot of the figure 2 of the supplementary data.
    k can take one of the following variable names: 'k1_bwd_relative', 'k1_fwd_relative', 'k2_bwd_relative', 'k1_bwd_relative'
    conc can take: 'E', 'ES', 'S', 'P'
    k_label and conc_label are the desired axes labels'''

    # prepare figure
    fg = sns.FacetGrid(data=data_k_conc, hue='volume_fraction', size = 4, aspect = 1.4,
                 palette= {0: 'b', 0.3: "#b54334", 0.5: "g"})
    ax = fg.axes[0][0]
    ax.set(xscale="log", yscale="log")

    fg.map(plt.scatter, conc, k).add_legend()

    # title and axis labels
    ax.set_xlabel(conc_label, fontsize = 12)
    ax.set_ylabel(k_label, fontsize = 12)
    return()
