"""
"""

import math

import numpy as np
import pandas as pd
import statsmodels.api as sm


# column numbers of important stuff KEEP UP TO DATE (better way to do this?)
FIPS = 0
COVID_DATA = range(1, 50)  # leaving out cols w/ less than half counties
LOCKDOWN = 99
POPULATION = 100
IND_DATA = range(101, 126)


def load_data(path):
    """
    """
    df = pd.read_csv(path)
    df["lockdown_delta"] = df["lockdown_delta"].fillna(0)
    df["lockdown_delta"] = df["lockdown_delta"] > 0
    df["lockdown_delta"] = df["lockdown_delta"].astype(int)

    return df


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

    # create count how many days each industry is significant for
    industry_counts = get_industry_counts(signif_vars)
    
    return models, industry_counts
        

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


    # create count how many days each industry is significant for
    industry_counts = get_industry_counts(signif_vars)
    
    return models, industry_counts


def get_industry_counts(signif_vars):
    """
    """
    reverse_signif = {}
    for day, industries in signif_vars.items():
        for industry in industries:
            reverse_signif[industry] = reverse_signif.get(industry, 0) + 1

    ordered_sig = [(k, v) for k, v in
        sorted(reverse_signif.items(),
            key=lambda item: item[1], reverse=True)]

    return ordered_sig


def get_sig_industries(model, alpha=0.05):
    """
    """
    pvals = model.get_robustcov_results().pvalues
    sig_vars = model.params.index[pvals < alpha]

    return sig_vars
