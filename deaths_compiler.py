#%%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import os


#%%
dataDir = "data/"
growth_rate = pd.read_csv(dataDir + "deaths_clean.csv", encoding="latin_1")
growth_rate


#%%
job_shares = pd.read_csv(dataDir + "jobs_shares.csv", encoding="latin_1")
num_fips = []
for fip in job_shares['geofips']:
    num_fips.append(int(fip.strip('"')))
job_shares['geofips'] = num_fips
job_shares = job_shares.rename(columns={"geofips" : 'fips'})


#%%
lockdown = pd.read_csv(dataDir + "lockdown_dates.csv")
us_lock = lockdown[lockdown['Country'] == 'United States']
state2fips = pd.read_csv(dataDir + "state_fips.csv").rename(columns={"stname" : "Place"})
state2fips["statefips"] = state2fips[' st'] * 1000
state2fips = state2fips.drop([' st', ' stusps'], axis = 1)

def export2csv(filedir, filename, data):
    if os.path.exists(filedir + filename):
        os.remove(filedir + filename)
        print("File deleted")
    else:
        print("The file does not exist")
    data.to_csv(filedir + filename, index=False, encoding='latin_1')

export2csv(dataDir, "state_fips_clean.csv", state2fips)


#%%
population = pd.read_csv(dataDir + "dollars_clean.csv", encoding = "latin_1")
population = population[['GeoFIPS', '20']]
population['20'] = population['20'].astype(int)
num_fips = []
for fip in population['GeoFIPS']:
    num_fips.append(int(fip.strip().strip('"')))
population['GeoFIPS'] = num_fips
population = population.rename(columns={"GeoFIPS" : 'fips', "20" : 'population'})

export2csv(dataDir, "population.csv", population)


#%%
fips_lockdown = pd.merge(us_lock, state2fips, on="Place")[['statefips', 'Start date']]
fips_lockdown = fips_lockdown.rename(columns={"Start date" : "lockdown start"})
fips_lockdown = fips_lockdown.sort_values('statefips')

export2csv(dataDir, "us_lockdown_dates.csv", fips_lockdown)


#%%
temp = pd.merge(growth_rate, fips_lockdown, on='statefips', how = 'outer')
temp['lockdown_delta'] = (pd.to_datetime(temp['lockdown start']) - pd.to_datetime(temp['start_date'])).dt.days
temp = pd.merge(temp, population, on='fips')
#temp[temp['fips'] == 53033].head(10)#


#%%
export = pd.merge(temp, job_shares, on='fips')
drop_cols = ['statefips', 'start_date', 'lockdown start']
for idx, col in enumerate(job_shares.columns):
    if idx < 9 and idx > 0:
        drop_cols.append(col)
export = export.drop(columns = drop_cols)

export2csv(dataDir, "deaths_regression_data.csv", export)


#%%
list(enumerate(export.columns))


#%%



