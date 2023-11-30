import pandas as pd 

## Loading/Extracting Data

# Landing Page: https://data.cityofnewyork.us/Public-Safety/NYPD-Arrest-Data-Year-to-Date-/uip8-fykc 
# Data Download Link: 
datalink = 'https://data.cityofnewyork.us/api/views/uip8-fykc/rows.csv?date=20231130&accessType=DOWNLOAD'

df = pd.read_csv(datalink)
df.size
df.sample(5)

df.columns




## Save data as csv to model_dev1/data/raw folder
df.to_csv('model_dev1/data/raw/arrest_data.csv', index=False)

## Save as pickle to model_dev1/data/raw folder folder
df.to_pickle('model_dev1/data/raw/arrest_data.pkl')