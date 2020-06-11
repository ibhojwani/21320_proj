"""
"""

import math

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
import statsmodels.api as sm
import tabulate


# column numbers of important stuff KEEP UP TO DATE (better way to do this?)
FIPS = 0
COVID_DATA = range(1, 50)  # leaving out cols w/ less than half counties
LAST_COVID = 98
LOCKDOWN = 99
POPULATION = 100
IND_DATA = range(101, 125)


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
    df = truncate_names(df)

    x_cols = [df.columns[POPULATION]] + list(df.columns[IND_DATA])
    reg_dat = df[x_cols]
    reg_dat = sm.add_constant(reg_dat)

    models, signif_vars, signif_coefs = run_regs(reg_dat, df, "Basic Model")

    coefs_df = get_mean_coefs(signif_coefs, signif_vars)

    save_coefs(coefs_df, "Basic Model")

    # create count how many days each industry is significant for
    industry_counts = get_industry_counts(signif_vars)
    
    return models, coefs_df
        

def model_lockdown(df):
    """
    """
    df = truncate_names(df)
    df[df.columns[LOCKDOWN]] = df[df.columns[LOCKDOWN]].fillna(-100)
    df[df.columns[LOCKDOWN]] = (df[df.columns[LOCKDOWN]] >= 0).astype(int)
    x_cols = [df.columns[LOCKDOWN]] + [df.columns[POPULATION]] + list(df.columns[IND_DATA])

    reg_dat = df[x_cols]
    reg_dat = sm.add_constant(reg_dat)

    models, signif_vars, signif_coefs = run_regs(reg_dat, df, "Lockdown Model")

    coefs_df = get_mean_coefs(signif_coefs, signif_vars)
    save_coefs(coefs_df, "Lockdown Model")

    # create count how many days each industry is significant for
    industry_counts = get_industry_counts(signif_vars)
    
    return models, coefs_df


def run_regs(reg_dat, df, mod_name):
    """
    """
    models = []
    signif_vars_d = {}
    signif_coefs_d = {}

    for day in COVID_DATA:
        reg_dat["Y"] = df[df.columns[day]] 
        reg_dat = reg_dat[~reg_dat["Y"].isna()]

        mod = sm.OLS(reg_dat["Y"], reg_dat.drop("Y", axis=1))
        results = mod.fit()
        signif_vars, signif_coefs = get_sig_industries(results)

        models.append(results)
        signif_vars_d[day] = signif_vars
        signif_coefs_d[day] = signif_coefs

        write_table(results, day, mod_name)

    return models, signif_vars_d, signif_coefs_d


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
    coefs = model.params
    sig_vars = model.params.index[pvals < alpha].values
    sig_coefs = coefs.loc()[sig_vars].values

    return sig_vars, sig_coefs


def get_mean_coefs(signif_coefs, signif_vars):
    """
    """
    industry_counts = {}
    industry_dict = {}
    rv = {}

    for day in IND_DATA:
        day = day - list(IND_DATA)[0]
        if not day in signif_vars.keys(): 
            continue
        for i, industry in enumerate(signif_vars[day]):
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
            industry_dict[industry] = industry_dict.get(industry, 0) + signif_coefs[day][i]
            if signif_coefs[day][i] > 2:
                print("howoh")
    
    for industry, coef_sum in industry_dict.items():
        rv[industry] = coef_sum / industry_counts[industry]

    rv_df = pd.DataFrame([pd.Series(rv), pd.Series(industry_counts)]).transpose()

    return rv_df


def save_coefs(coef_df, name):
    print(coef_df)
    with open("summary_tables/{}.tex".format(name), "w") as f:
        f.write(tabulate.tabulate(coef_df, tablefmt="latex"))
    
    return None


def write_table(mod, day_num, mod_name):
    """
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
    """
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
    """
    """
    coefs_df_base = model_1(df)[1]
    coefs_df_base.columns = ["(1) Avg Signif. Coef", "(1) num days signif."]

    coefs_df_lockdown = model_lockdown(df)[1]
    coefs_df_lockdown.columns = ["(2) Avg Signif. Coef", "(2) num days signif."]

    joined_df = coefs_df_base.merge(coefs_df_lockdown, how="outer", right_on=coefs_df_lockdown.index, left_on=coefs_df_base.index)
    joined_df = joined_df.rename(columns={"key_0": "Industry"})
    joined_df.set_index("Industry")
    joined_df = joined_df.sort_values("(1) num days signif.", ascending=False)
    joined_df = joined_df.drop("federal civilian")
    with open("summary_tables/joined.tex", "w") as f:
        # title = r"\begin{center}\textbf{Average coefficients on significant days, and number of significant days, per industry}\end{center}" 
        # f.write(title)
        f.write(tabulate.tabulate(joined_df, tablefmt="latex", headers="keys", showindex="never"))

    return joined_df

