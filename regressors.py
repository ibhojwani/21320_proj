#%%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm

import os


#%%
dataDir = "data/"
df = pd.read_csv(dataDir + "jobs.csv", encoding="latin_1")
for idx, sector in enumerate(df['Description']):
    if sector != sector:
        df = df.drop(idx)


#%%
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


#%%
data = data[data['total employment (number of jobs)'] != 0]
data


#%%
if os.path.exists(dataDir + "jobs_clean.csv"):
    os.remove(dataDir + "jobs_clean.csv")
    print("File deleted")
else:
    print("The file does not exist")
data.to_csv(dataDir + 'jobs_clean.csv', index=False, encoding='latin_1')


#%%
job_numbers = data.drop(labels = 'geofips', axis = 1).to_numpy()
total, subtotals = np.split(job_numbers, [1], axis=1)
geofips = data['geofips'].to_numpy().reshape((-1, 1))


#%%



#%%
sector_shares = np.hstack((geofips, total, subtotals / total))
sector_shares = pd.DataFrame(sector_shares, columns = col_names)
sector_shares

if os.path.exists(dataDir + "jobs_shares.csv"):
    os.remove(dataDir + "jobs_shares.csv")
    print("File deleted")
else:
    print("The file does not exist")
sector_shares.to_csv(dataDir + 'jobs_shares.csv', index=False, encoding='latin_1')

#%% [markdown]
# 218 is Monterey, CA (large ass aquarium and fishing disctrict, perfectly reasonable)
# 
# 355 is Hendry, FL (which is home to the Okaloacoochee Slough State Forest and various state preserves)
# 
# 2462 is Sully, SD (right next to the Missouri River, so I guess tons of fishing)
# 
# 2722 is McMullen, TX (apparently richest county in Texas. Very small)
# 
# The only one that remotely makes sence is Monterey...

#%%



#%%


#%% [markdown]
# Publishing includes software publishing (so here's "big tech", so to speak)
# 
# 06075 is San Francisco, CA (omg what a shocker)
# 
# 06081 is San Mateo, CA (Sony is headquartered here)
# 
# 06085 is Santa Clara, CA (wow, I wonder what big tech companies are here)
# 
# 08014 is Broomfield, CO (apparently a telecomm company named Lvl 3 Communications. Also a small company names Oracle)
# 
# 13121 is Fulton Country, GA (it's Atlanta. Too many possibilities to count)
# 
# 36061 is New York, NY (I'm, uh, not gonna bother on this one)
# 
# 53033 is King County WA (This is home turf. Amazon, Microsoft, Steam, all here)

#%%



#%%



