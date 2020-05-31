"""
"""

import math

import numpy as np
import pandas as pd
import statsmodels.api as sm


# column numbers of important stuff KEEP UP TO DATE (better way to do this?)
FIPS = 0
COVID_DATA = range(1, 50)  # leaving out cols w/ less than half counties
LOCKDOWN = 85
POPULATION = 86
IND_DATA = range(87, 112)


def model_1(df):
    """ basic model with industry and pop counts, no lockdown
    """
    models = []
    signif_vars = {}

    x_cols = [df.columns[POPULATION]] + list(df.columns[IND_DATA])
    reg_dat = df[x_cols]
    reg_dat = sm.add_constant(reg_dat)

    # build model for each day
    for day in COVID_DATA:
        reg_dat["Y"] = df[df.columns[day]] 
        reg_dat = reg_dat[~reg_dat["Y"].isna()]
        mod = sm.OLS(reg_dat["Y"], reg_dat.drop("Y", axis=1))

        results = mod.fit()
        models.append(results)
        signif_vars[day] = get_sig_industries(results)

    reverse_signif = {}

    # create count how many days each industry is significant for
    for day, industries in signif_vars.items():
        for industry in industries:
            reverse_signif[industry] = reverse_signif.get(industry, 0) + 1
        
    return models, reverse_signif


def model_lockdown(df):
    """
    """
    models = []
    signif_vars = {}

    x_cols = [df.columns[LOCKDOWN]] + [df.columns[POPULATION]] + list(df.columns[IND_DATA])
    reg_dat = df[x_cols]
    reg_dat = sm.add_constant(reg_dat)

    # build model for each day
    for day in COVID_DATA:
        reg_dat["Y"] = df[df.columns[day]] 
        reg_dat = reg_dat[~reg_dat["Y"].isna()]
        mod = sm.OLS(reg_dat["Y"], reg_dat.drop("Y", axis=1))

        results = mod.fit()
        models.append(results)
        signif_vars[day] = get_sig_industries(results)

    reverse_signif = {}

    # create count how many days each industry is significant for
    for day, industries in signif_vars.items():
        for industry in industries:
            reverse_signif[industry] = reverse_signif.get(industry, 0) + 1
        
    return models, reverse_signif

def get_sig_industries(model, alpha=0.05):
    """
    """
    pvals = model.get_robustcov_results().pvalues
    sig_vars = model.params.index[pvals < alpha]

    return sig_vars
