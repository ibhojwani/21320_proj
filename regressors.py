import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import os

dataDir = "data/"
df = pd.read_csv(dataDir + "jobs.csv", encoding="latin_1")
for idx, sector in enumerate(df['Description']):
    if sector != sector:
        df = df.drop(idx)

sectors = df['Description'].unique().tolist()
for idx, sector in enumerate(tqdm(sectors)):
    sectors[idx] = sector.strip().lower()
col_names = np.insert(sectors, 0, ['geofips'])
data = pd.DataFrame(columns=col_names)

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

data = data[data['total employment (number of jobs)'] != 0]

if os.path.exists(dataDir + "jobs_clean.csv"):
    os.remove(dataDir + "jobs_clean.csv")
    print("File deleted")
else:
    print("The file does not exist")
data.to_csv(dataDir + 'jobs_clean.csv', index=False, encoding='latin_1')

job_numbers = data.drop(labels = 'geofips', axis = 1).to_numpy()
total, subtotals = np.split(job_numbers, [1], axis=1)
geofips = data['geofips'].to_numpy().reshape((-1, 1))

sector_shares = np.hstack((geofips, total, subtotals / total))
sector_shares = pd.DataFrame(sector_shares, columns = col_names)

if os.path.exists(dataDir + "jobs_shares.csv"):
    os.remove(dataDir + "jobs_shares.csv")
    print("File deleted")
else:
    print("The file does not exist")
sector_shares.to_csv(dataDir + 'jobs_shares.csv', index=False, encoding='latin_1')