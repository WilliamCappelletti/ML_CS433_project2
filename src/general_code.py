# -*- coding: utf-8 -*-
'''Shared code that was used for data analysis.

    The common code is accessible at the module level, while regression specific code is stored in User-specific modules.
    The users are charles, marie, william

    example:
        import general_code

        general_code.william.module_test() // return 0'''

import william, charles, marie
import numpy as np
import pandas as pd
import statsmodels.api as statsmodels
from pandas import DataFrame

def cast_categories(frame, column, cat_name = 'Group'):
    '''Returns a new data frame in which the specified column has been changed by new colums representing the appertenance to one specified category'''
    
    new_frame = frame.copy()
    categories = new_frame[column].unique()
    splitted_cols = {'{cat_name}_{x}'.format(cat_name=cat_name, x=x): [ 1 if inside else 0 for inside in new_frame[column]==x] for x in categories}
    splitted_cols = pd.DataFrame(splitted_cols, index = new_frame.index)
    new_frame = pd.concat([new_frame.drop(columns=[column]), splitted_cols], axis=1)
    return new_frame
