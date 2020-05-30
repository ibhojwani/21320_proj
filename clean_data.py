"""
NOTES:
01001 day 36 (5/14) loses a case even after adding deaths. min log diff is -1.2
gets close to 0 quickly
"""

import numpy as np
import pandas as pd


DATA_DIR = "data/"
COVID_PATH = "data/covid_raw.csv"
INDUSTRY_PATH = "data/industry_raw.csv"

# number of cases to mark the beginning of the outbreak in a county
STARTING_CASES = 15

# Set it to None to display all columns in the dataframe
pd.set_option('display.max_columns', None)


def clean_covid(outfile=""):
    """
    """
    df = pd.read_csv(COVID_PATH, dtype={"fips": str})

    # Only keep fips, date, and cases
    df["cases"] = df["cases"] + df["deaths"]
    df = df.drop(df.columns[5:], axis=1)
    df = df.drop(["county", "state"], axis=1)

    # pivot rows to get panel data
    df = df.pivot_table(index="fips", columns="date", values="cases")

    # left align and drop values below threshold
    df = df.fillna(0)
    df = df.apply(realign_covid_row, axis=1)

    # drop cols/rows with all nan
    df = df.dropna(how="all", axis=1)
    df = df.dropna(how="all", axis=0)

    # Get daily growth rates
    df = np.log(df)
    df = df.diff(axis=1)

    # # deal with persistant negative growth rates w/ interpolation
    # df = df.fillna(np.infty)
    # df[df < 0] = np.nan
    # df = df.interpolate(axis=1)

    df[df == np.infty] = np.nan

    df["statefips"] = df.index.str[0:2] + "000"

    # rename columns
    df_case_cols = ["day_" + str(i) for i in range(len(df.columns) - 1)]
    df.columns = df_case_cols + ["statefips"]

    # taking difference sets day 0 to nan
    df = df.drop("day_0", axis=1)  

    if outfile:
        df.to_csv(DATA_DIR + outfile, index=True)
    return df


def realign_covid_row(row):
    """
    """
    # left align data
    zero_count = 0
    for i in row:
        if i < STARTING_CASES:
            zero_count += 1
        else:
            break
    row[:] = list(row[zero_count:]) + [np.nan] * zero_count

    # add back statefips
    return row
