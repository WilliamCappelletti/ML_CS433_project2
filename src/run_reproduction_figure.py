'''
script to reproduce the plots in the original paper

Generate the big plot, possible to plot the individual plots by using the function  plot_fig2_subplot() in figure_generation.py
'''

import figure_generation

#path to the data file
file = "../Data/result_full_factorial_pgm.csv"


#plots and save the figure
figure_generation.plot_fig2_plot(file, save = True) #comment to avoid saving it again
