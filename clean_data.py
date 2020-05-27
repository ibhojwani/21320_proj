"""
NOTES:
Some counties have no data. check which these are in case they will induce bias
Some counties have some missing data. check which for bias.
"""

import numpy as np
import pandas as pd


DATA_DIR = "data/"
COVID_PATH = "data/COVID-19_Historical_Data_Table.csv"
INDUSTRY_PATH = "data/CAINC5N__ALL_AREAS_2001_2018.csv"

# Set it to None to display all columns in the dataframe
pd.set_option('display.max_columns', None)


def clean_covid():
    """
    """
    df = pd.read_csv(COVID_PATH)

    df = df[df["GEO"] == "County"]
    df_state = df[df["GEO"] == "State"]

    return df


def clean_industry(outfile=""):
    """
    """
    df = pd.read_csv(INDUSTRY_PATH, encoding = "latin")

    # Drop earlier years
    drop_cols = df.columns[8:25]
    df = df.drop(drop_cols, axis=1)  # drop earlier years
    df = df.drop(["TableName", "Region", "GeoName",
                "IndustryClassification", "Description"], axis=1)

    # Drop non-data rows
    df = df.iloc[0:-4]

    # Drop rows w/ subindustry info
    all_codes = np.array(pd.unique(df["LineCode"]))
    keep_codes = all_codes[all_codes < 100]
    keep_codes = np.append(keep_codes, np.arange(100, 2000, 100))
    keep_codes  = np.append(keep_codes, all_codes[all_codes >= 2000])

    df = df[df["LineCode"].isin(keep_codes)]

    # Unit column replacement / apply unit multiplier
    df["2018"] = pd.to_numeric(df["2018"], errors="coerce")
    unit_dict = {"Thousands of dollars": 1000,
                "Dollars": 1,
                "Number of persons": 1}

    df["Unit"] = df["Unit"].replace(unit_dict)
    df["Value"] = df["Unit"] * df["2018"]

    df = df.drop(["Unit", "2018"], axis=1)

    # Final cleaning
    df = df.pivot(columns="LineCode", values="Value")  # pivot codes to cols
    df.columns = [str(round(col)) for col in df.columns]  # col names to str
    df = df.loc[~df["10"].isna()]  # drop counties with empty data
    
    # to file
    if outfile:
        df.to_csv(DATA_DIR + outfile, index=False)

    return df