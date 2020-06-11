"""
"""


from matplotlib import pyplot as plt
import seaborn as sbn
import regressions

import numpy as np
import pandas as pd


def two_ind_viz():
    """ Compares tseries of average growth rates of top manufacturing and information counties
    """
    df = regressions.load_data("data/regression_data.csv") 

    # get information counties
    top_info = df.sort_values("information", ascending=False)["information"]
    top_info = df.loc()[top_info.index[0:5]]

    # get manufacuring counties
    top_manu = df.sort_values("manufacturing", ascending=False)["manufacturing"] 
    top_manu = df.loc()[top_manu.index[0:5]]

    # take first 40 days
    top_info1 = top_info[top_info.columns[1:40]] 
    top_manu1 = top_manu[top_manu.columns[1:40]] 

    # control for population
    temp_dict = {
        "Information": top_info1.mean() /top_info["population"].mean(),
        "Manufacturing": top_manu1.mean()/top_manu["population"].mean()}
    final_df = pd.DataFrame(temp_dict)

    # plot
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
    """ Plots tseries of coefficients over time of top significant industries from the base model
    """
    # top industries
    target_inds_l = ["management of companies and en...",
        "administrative and support and...",
        "transportation and warehousing",
        "educational services",
        "wholesale trade"]
    target_inds = {k: [] for k in target_inds_l}

    # get coefficients from regression
    df = regressions.load_data("data/regression_data.csv")
    models, rob, sig = regressions.model_1(df)
    sig_df = pd.DataFrame(sig).drop(sig.columns[-1], axis=1)
    sig_counts = sig_df.loc()[target_inds_l]

    # pull coefficients from each model for each industry/day
    for model in models:
        for ind, val in model.params[target_inds].items():
            target_inds[ind] += [val]
        
    inds_df = pd.DataFrame(target_inds)
    
    plt.close()
    sbn.lineplot(data=inds_df)
    sbn.lineplot([0, 49], [0, 0], color='k', linestyle='-', linewidth=2)
    plt.ylabel("Coefficient")
    plt.xlabel("Day")
    plt.title("Affects of some industries on COVID spread (Base)")

    inds_df = inds_df.apply(keep_largest, axis=0,  args=[sig_counts])
    sbn.scatterplot(data=inds_df, s=100)

    return target_inds


def time_series_top_lockdown():
    """ Plots tseries of coefficients over time of top significant industries from the lockdown model
    """
    target_inds_l = ["management of companies and en...",
        "administrative and support and...",
        "transportation and warehousing",
        "manufacturing",
        "wholesale trade"]
    target_inds = {k: [] for k in target_inds_l}

    df = regressions.load_data("data/regression_data.csv")
    models, rob, sig = regressions.model_lockdown(df)
    sig_df = pd.DataFrame(sig).drop(sig.columns[-1], axis=1)
    print(sig_df)
    sig_counts = sig_df.loc()[target_inds_l]

    for model in models:
        for ind, val in model.params[target_inds].items():
            target_inds[ind] += [val]
        
    inds_df = pd.DataFrame(target_inds)
    
    plt.close()
    sbn.lineplot(data=inds_df)
    sbn.lineplot([0, 49], [0, 0], color='k', linestyle='-', linewidth=2)
    plt.ylabel("Coefficient")
    plt.xlabel("Day")
    plt.title("Affects of some industries on COVID spread (Lockdown)")

    inds_df = inds_df.apply(keep_largest, axis=0,  args=[sig_counts])
    sbn.scatterplot(data=inds_df)

    return target_inds



def keep_largest(col, sig_counts):
    """ helper to get significant days so we can make a scatterplot of just the significant days
    """
    # sort and set all values below alpha significance cutoff to NaN (so they wont be plotted)
    ordered = sorted(np.abs(col), reverse=True)
    cutoff = ordered[int(sig_counts.loc()[col.name][1])]
    col[np.abs(col) < cutoff] = np.nan

    return col


def get_day_counts(df, covid_range):
    """ Histogram of the length of outbreak in each county (till present)
    """
    plt.close()
    num_days = (~df[df.columns[covid_range]].isna()).astype(int).sum(axis=1).to_numpy()

    counts = []
    for i in range(min(num_days) - 1, max(num_days) + 1):
        counts.append(sum((num_days >= i)))
    
    sbn.lineplot(x=range(min(num_days)-1, max(num_days)+1), y=counts)
    plt.title("Number of Counties with Outbreaks of Given Length")
    plt.xlabel("Length of outbreak")
    plt.ylabel("Number of counties")
    plt.plot([50, 50], [0, 2500], linewidth=2, color="black")


    return None


def get_lag_diffs(df):
    """ Plots average time it took for counties to hit certain numbers of people sick
    """
    plt.close()
    datas = []
    df = df[df.columns[39:]]
    df = df.fillna(0)

    for i in range(1, 21):
        df["count"] = ((df > 0) & (df <= i)).astype(int).sum(axis=1)
        datas.append(df["count"].mean())
    
    sbn.lineplot(x=range(1, 21), y=datas)
    plt.xlabel("Minimum number of cases threshold")
    plt.ylabel("Average days till threshold hit")
    plt.title("Average number of days for counties to hit a number of cases")
    plt.plot([6, 6], [8, 21], linewidth=2, color="black")

    plt.savefig("figs/lags.png")