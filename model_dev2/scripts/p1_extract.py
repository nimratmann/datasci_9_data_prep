import pandas as pd 

## Loading/Extracting Data

# Landing Page: https://data.cityofnewyork.us/Public-Safety/NYPD-Hate-Crimes/bqiq-cu78
# Data Download Link: 
datalink = 'https://data.cityofnewyork.us/api/views/bqiq-cu78/rows.csv?date=20231130&accessType=DOWNLOAD'

df = pd.read_csv(datalink)
df.shape
df.sample(5)

df.columns


## Saving data as a csv to model_dev1/data/raw folder
df.to_csv('model_dev2/data/raw/hate_crimes.csv', index=False)

## Saving as pickle to model_dev1/data/raw folder folder
df.to_pickle('model_dev2/data/raw/hate_crimes.pkl')