"""
"""

import math

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
import statsmodels.api as sm
import tabulate


# column numbers of important things in df (better way to do this?)
FIPS = 0
COVID_DATA = range(1, 50)  # leaving out cols w/ less than half counties
LAST_COVID = 98
LOCKDOWN = 99
POPULATION = 100
IND_DATA = range(101, 125)


def load_data(path):
    """ Loads preprocessed datafile and conforms it as required
    """
    df = pd.read_csv(path)
    df["lockdown_delta"] = df["lockdown_delta"].fillna(0)
    df["lockdown_delta"] = df["lockdown_delta"] > 0
    df["lockdown_delta"] = df["lockdown_delta"].astype(int)

    return df


def model_1(df):
    """ basic model with industry and pop counts, no lockdown
    """
    # change col names for ease of printing in table
    df = truncate_names(df)

    # create x variables using relevant columns
    x_cols = [df.columns[POPULATION]] + list(df.columns[IND_DATA])
    reg_dat = df[x_cols]
    reg_dat = sm.add_constant(reg_dat)

    # run regressions and get regr and robustness outputs
    models, robust, signif_vars, signif_coefs, rob_vars = run_regs(reg_dat, df, "Basic Model")

    # compile information on what industries were significant
    coefs_df = get_mean_coefs(signif_coefs, signif_vars, rob_vars)

    # save results to file
    save_coefs(coefs_df, "Basic Model")

    return models, robust, coefs_df
        

def model_lockdown(df):
    """ Basic model + lockdown indicator
    """
    df = truncate_names(df)

    # prep lockdown columns
    df[df.columns[LOCKDOWN]] = df[df.columns[LOCKDOWN]].fillna(-100)
    df[df.columns[LOCKDOWN]] = (df[df.columns[LOCKDOWN]] >= 0).astype(int)

    # x vars
    x_cols = [df.columns[LOCKDOWN]] + [df.columns[POPULATION]] + list(df.columns[IND_DATA])
    reg_dat = df[x_cols]
    reg_dat = sm.add_constant(reg_dat)

    # run model and get results + robustness
    models, robust, signif_vars, signif_coefs, rob_vars  = run_regs(reg_dat, df, "Lockdown Model")

    # compile significant industry data and save
    coefs_df = get_mean_coefs(signif_coefs, signif_vars, rob_vars)
    save_coefs(coefs_df, "Lockdown Model")
    
    return models, robust, coefs_df


def run_regs(reg_dat, df, mod_name):
    """ Given X and Y data, run regressions for each day

    Inputs:
        reg_dat: X data
        df: original data
        mod_name: for naming output file
    
    Returns:
        models: list of sm.OLS model objects w/ regression for each day
        robust: list of sm.OLS model objects w/ robustness results for each day
        signif_vars_d: dict w/ list of significant X vars for each day
        signif_coefs_d: dict w/ lsit of coefficients for each signif var each day
        rob_vars_d: dict w/ list of signif vars that passed robustness check each day
    """
    # init output vars
    robust = []
    models = []
    signif_vars_d = {}
    signif_coefs_d = {}
    rob_vars_d = {}

    for day in COVID_DATA:
        # build Y data and drop empty rows for that day
        reg_dat["Y"] = df[df.columns[day]] 
        reg_dat = reg_dat[~reg_dat["Y"].isna()]
        Y = reg_dat["Y"]
        reg_dat = reg_dat.drop("Y", axis=1)

        # regress and do robust check on significant vars, and store list of
        # signif industires
        mod = sm.OLS(Y, reg_dat)
        results = mod.fit()
        signif_vars, signif_coefs = get_sig_industries(results)

        temp_dat = reg_dat[signif_vars]
        mod = sm.OLS(Y, temp_dat)
        rob_res = mod.fit()
        rob_vars, rob_coefs = get_sig_industries(rob_res)

        # store results
        robust.append(rob_res)
        models.append(results)

        signif_vars_d[day] = signif_vars
        signif_coefs_d[day] = signif_coefs

        rob_vars_d[day] = rob_vars

        write_table(results, day, mod_name)
        write_table(rob_res, day, mod_name + "robustness")

    return models, robust, signif_vars_d, signif_coefs_d, rob_vars_d


def get_sig_industries(model, alpha=0.05):
    """ Given sm.OLS output, gets names and coefs of significant X variables
    """
    pvals = model.get_robustcov_results().pvalues
    coefs = model.params
    sig_vars = model.params.index[pvals < alpha].values
    sig_coefs = coefs.loc()[sig_vars].values

    return sig_vars, sig_coefs


def get_mean_coefs(signif_coefs, signif_vars, rob_vars):
    """ Compiles x var significance data across all days into summary dataframe.
    Returns the number of days each var was significant for, and the number
    of days it passed the robustness check.
    """
    industry_counts = {}
    industry_dict = {}
    rob_counts = {}
    rv = {}

    for day in IND_DATA:
        day = day - list(IND_DATA)[0]
        if not day in signif_vars.keys(): 
            continue
        
        # get number of days of significance
        for i, industry in enumerate(signif_vars[day]):
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
            # industry_dict[industry] = industry_dict.get(industry, 0) + signif_coefs[day][i]
        for i, industry in enumerate(rob_vars[day]):
            rob_counts[industry] = rob_counts.get(industry, 0) + 1

    # calc average coeff across all days
    # for industry, coef_sum in industry_dict.items():
    #     rv[industry] = coef_sum / industry_counts[industry]

    rv_df = pd.DataFrame([pd.Series(industry_counts), pd.Series(rob_counts)]).transpose()

    return rv_df


def save_coefs(coef_df, name):
    """ Saves summary table from regressions in latex
    """
    with open("summary_tables/{}.tex".format(name), "w") as f:
        f.write(tabulate.tabulate(coef_df, tablefmt="latex"))
    
    return None


def write_table(mod, day_num, mod_name):
    """ Saves regression outputs for a given day in latex
    """
    file_name = "regression_outputs/{}_{}.tex".format(mod_name, day_num).replace(" ", "")
    with open(file_name, "w") as f:
        f.write(r"\documentclass[12pt]{article} \usepackage{booktabs}\begin{document}")
        output = mod.get_robustcov_results().summary().tables[1].as_latex_tabular()
        title = r"\begin{center}\large{Day " + str(day_num) + " " + mod_name + r"}\end{center}" 
        f.write(title + "\n\n")
        f.write(output)
        f.write(r"\end{document}")


def truncate_names(df):
    """ Truncates column names so it is easier to print in table
    """
    col_names = df.columns[IND_DATA]
    new_names = []
    for name in col_names:
        if len(name) > 30:
            new_names.append(name[:30] + "...")
        else:
            new_names.append(name)
    df = df.rename(columns = {k: v for k, v in zip(col_names, new_names)})

    return df


def both_models(df):
    """ Runs both models and performs all tasks, building final summary tables
    """
    coefs_df_base = model_1(df)[2]
    coefs_df_base.columns = ["(1) num days signif.", "(1) num days robust"]

    coefs_df_lockdown = model_lockdown(df)[2]
    coefs_df_lockdown.columns = ["(2) num days signif.", "(2) num days robust"]

    # joins both regressions into final summary table
    joined_df = coefs_df_base.merge(coefs_df_lockdown, how="outer", right_on=coefs_df_lockdown.index, left_on=coefs_df_base.index)
    joined_df = joined_df.rename(columns={"key_0": "Industry"})
    joined_df.set_index("Industry")
    joined_df = joined_df.sort_values("(1) num days signif.", ascending=False)
    # joined_df = joined_df.drop("federal civilian")

    # saves table
    with open("summary_tables/joined.tex", "w") as f:
        # title = r"\begin{center}\textbf{Average coefficients on significant days, and number of significant days, per industry}\end{center}" 
        # f.write(title)
        f.write(tabulate.tabulate(joined_df, tablefmt="latex", headers="keys", showindex="never"))

    return joined_df

