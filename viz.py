"""
"""


from matplotlib import pyplot as plt
import seaborn as sbn
import regressions

import numpy as np
import pandas as pd


def viz():

    df = regressions.load_data("data/regression_data.csv") 

    top_info = df.sort_values("information", ascending=False)["information"]
    top_info = df.loc()[top_info.index[0:5]]

    top_manu = df.sort_values("manufacturing", ascending=False)["manufacturing"] 
    top_manu = df.loc()[top_manu.index[0:5]]

    top_info1 = top_info[top_info.columns[1:40]] 
    top_manu1 = top_manu[top_manu.columns[1:40]] 
    temp_dict = {
        "Information": top_info1.mean() /top_info["population"].mean(),
        "Manufacturing": top_manu1.mean()/top_manu["population"].mean()}
    final_df = pd.DataFrame(temp_dict)

    plt.close()
    sbn.lineplot(data=final_df)
    plt.axes().set_xticks(np.arange(1, 40, 5))
    plt.axes().set_xticklabels(np.arange(1, 40, 5))
    sbn.lineplot([0, 40], [0, 0], color='k', linestyle='-', linewidth=2)

    plt.xlabel("Day")
    plt.ylabel("Adjusted growth")
    plt.title("Average difference from mean growth by 2 industries, adjusted for population")

    return None


def time_series_top():
    """
    """
    target_inds = ["manufacturing",
        "administrative and support and waste management and remediation services",
        "transportation and warehousing",
        "mining, quarrying, and oil and gas extraction",
        "wholesale trade"]
    target_inds = {k: [] for k in target_inds}

    df = regressions.load_data("data/regression_data.csv")
    models, sig = regressions.model_1(df)
    sig_df = pd.DataFrame(sig).set_index(0)
    sig_counts = sig_df.loc()[target_inds]

    for model in models:
        for ind, val in model.params[target_inds].items():
            target_inds[ind] += [val]
        
    inds_df = pd.DataFrame(target_inds)
    
    plt.close()
    sbn.lineplot(data=inds_df)
    sbn.lineplot([0, 49], [0, 0], color='k', linestyle='-', linewidth=2)
    plt.ylabel("Coefficient")
    plt.xlabel("Day")
    plt.title("Affects of some industries on COVID spread")

    inds_df = inds_df.apply(keep_largest, axis=0,  args=[sig_counts])
    sbn.scatterplot(data=inds_df, s=100)

    return target_inds


def time_series_top_lockdown():
    """
    """
    target_inds = ["manufacturing",
        "administrative and support and waste management and remediation services",
        "transportation and warehousing",
        "mining, quarrying, and oil and gas extraction",
        "wholesale trade"]
    target_inds = {k: [] for k in target_inds}

    df = regressions.load_data("data/regression_data.csv")
    models, sig = regressions.model_lockdown(df)
    sig_df = pd.DataFrame(sig).set_index(0)
    sig_counts = sig_df.loc()[target_inds]

    for model in models:
        for ind, val in model.params[target_inds].items():
            target_inds[ind] += [val]
        
    inds_df = pd.DataFrame(target_inds)
    
    plt.close()
    sbn.lineplot(data=inds_df)
    sbn.lineplot([0, 49], [0, 0], color='k', linestyle='-', linewidth=2)
    plt.ylabel("Coefficient")
    plt.xlabel("Day")
    plt.title("Affects of some industries on COVID spread")

    inds_df = inds_df.apply(keep_largest, axis=0,  args=[sig_counts])
    sbn.scatterplot(data=inds_df)

    return target_inds



def keep_largest(col, sig_counts):
    """ helper to get top days in pandas
    """
    ordered = sorted(np.abs(col), reverse=True)
    cutoff = ordered[sig_counts.loc()[col.name][1]]

    col[np.abs(col) < cutoff] = np.nan

    return col

