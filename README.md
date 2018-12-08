# ML_project_2
second project for the course of machine learning CS433 in collaboration with the Laboratory of Computational Systems Biotechnology (SB/SV/STI) LCSB, based on the article "Particle-based simulation reveals macromolecular crowding effects on the Michaelis-Menten mechanism", by Weilandt, Daniel R and Hatzimanikatis, Vassily.

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

## References:

Particle-based simulation reveals macromolecular crowding effects on the Michaelis-Menten mechanism                                    
Daniel R Weilandt, Vassily Hatzimanikatis                                                                                           
doi: https://doi.org/10.1101/429316 

Seabold, Skipper, and Josef Perktold. “Statsmodels: Econometric and statistical modeling with python.” Proceedings of the 9th Python in Science Conference. 2010.


## Links 

overleaf report: https://www.overleaf.com/9168994853jnrbvbtpzrsf
