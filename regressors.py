import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from tqdm import tqdm

import os

# Loads the jobs data
dataDir = "data/"
df = pd.read_csv(dataDir + "jobs_raw.csv", encoding="latin_1")
for idx, sector in enumerate(df['Description']):
    if sector != sector:
        df = df.drop(idx)

# Creates a blank data frame with a column for every industry type
sectors = df['Description'].unique().tolist()
for idx, sector in enumerate(tqdm(sectors)):
    sectors[idx] = sector.strip().lower()
col_names = np.insert(sectors, 0, ['geofips'])
data = pd.DataFrame(columns=col_names)

''' This block of code gathers the 2018 industry data, cleans null
values, and then fills our blank data frame with job data for each
county.
'''
df = df[['GeoFIPS', 'Description', '2018']]
fips = df['GeoFIPS'].unique()
a = {}
for fip in fips:
    a[fip.strip()] = []
arr = df.to_numpy()
for v in tqdm(arr):
    if (v[2] == "(D)" or v[2] == "(NA)" or v[2] == "(NM)"):
        v[2] = 0
    a[v[0].strip()].append((v[1].strip().lower(), int(v[2])))
for fip in tqdm(a.keys()):
    new_row = {}
    new_row['geofips'] = fip
    for pair in a[fip]:
        sector, jobs = pair
        new_row[sector] = jobs
    data = data.append(new_row, ignore_index=True)

# Some counties exist with no employment data. We remove these
data = data[data['total employment (number of jobs)'] != 0]
# Removes the other sector to remove multicolinearity
data = data.drop(columns=['other services (except government and government enterprises)'])

# csv export for raw job numbers
if os.path.exists(dataDir + "jobs_clean.csv"):
    os.remove(dataDir + "jobs_clean.csv")
    print("File deleted")
else:
    print("The file does not exist")
data.to_csv(dataDir + 'jobs_clean.csv', index=False, encoding='latin_1')

# We extract the numerical data and the job totals to prepare the share calculation
job_numbers = data.drop(labels = 'geofips', axis = 1).to_numpy()
total, subtotals = np.split(job_numbers, [1], axis=1)
geofips = data['geofips'].to_numpy().reshape((-1, 1))

# We calculate the shares using vector operations for efficiency
sector_shares = np.hstack((geofips, total, subtotals / total))
col_names = data.columns
sector_shares = pd.DataFrame(sector_shares, columns = col_names)

# csv export for job shares
if os.path.exists(dataDir + "jobs_shares.csv"):
    os.remove(dataDir + "jobs_shares.csv")
    print("File deleted")
else:
    print("The file does not exist")
sector_shares.to_csv(dataDir + 'jobs_shares.csv', index=False, encoding='latin_1')