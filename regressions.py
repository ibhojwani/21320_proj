"""
"""

import math

import numpy as np
import pandas as pd
import statsmodels.api as sm


# column numbers of important stuff KEEP UP TO DATE (better way to do this?)
FIPS = 0
COVID_DATA = range(1, 85)
POPULATION = 87
IND_DATA = range(88, 112)


def model_1(df):
    """
    """
    models = []

    x_cols = [df.columns[POPULATION]] + list(df.columns[IND_DATA])
    reg_dat = df[x_cols]
    reg_dat = sm.add_constant(reg_dat)

    for day in COVID_DATA:
        print(day)
        
        reg_dat["Y"] = df[df.columns[day]]
        reg_dat = reg_dat[~reg_dat["Y"].isna()]
        mod = sm.OLS(reg_dat["Y"], reg_dat.drop("Y", axis=1))
        results = mod.fit()
        models.append(results)

    return models
