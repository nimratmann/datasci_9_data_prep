import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder

# Loading the Raw Pickle Data
df = pd.read_pickle('model_dev1/data/raw/arrest_data.pkl')

## Getting column names
df.columns

## Cleaning column names 
## Making them all lower case and removing white spaces
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns

## Getting data types
df.dtypes 

## drop columns
to_drop = [
    'arrest_key',
    'pd_cd',
    'pd_desc',
    'ky_cd',
    'law_code',
    'law_cat_cd',
    'jurisdiction_code',
    'x_coord_cd',
    'y_coord_cd',
    'latitude',
    'longitude',
    'new_georeferenced_column',
    ]
df.drop(to_drop, axis=1, inplace=True, errors='ignore')

#######################
## Changing arrest_date column from mm/dd/yyyy format to day of the week 
df['arrest_date'] = pd.to_datetime(df['arrest_date'])
df['arrest_date'] = df['arrest_date'].dt.day_name()

# Performing ordinal encoding on arrest_date column. 1-Monday, 2-Tuesday....7-Sunday. 
day_to_number = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
df['arrest_date'] = df['arrest_date'].replace(day_to_number).astype(int)

# Creating dataframe with mapping
df_mapping_date = pd.DataFrame({'arrest_date': list(day_to_number.keys()), 'arrest_date_num': list(day_to_number.values())})
df_mapping_date

# Saving mapping to csv
df_mapping_date.to_csv('model_dev1/data/processed/mapping_date.csv', index=False)

############

## Encoding the ofns_desc column
enc = OrdinalEncoder()
enc.fit(df[['ofns_desc']])
df['ofns_desc'] = enc.transform(df[['ofns_desc']])

# Creating a dataframe with mapping
df_mapping_offense = pd.DataFrame(enc.categories_[0], columns=['ofns_desc'])
df_mapping_offense['offense_ordinal'] = df_mapping_offense.index
df_mapping_offense.head(5)
# save mapping to csv
df_mapping_offense.to_csv('model_dev1/data/processed/mapping_offense.csv', index=False)

############

## Encoding the arrest_boro column
enc = OrdinalEncoder()
enc.fit(df[['arrest_boro']])
df['arrest_boro'] = enc.transform(df[['arrest_boro']])

# Creating a dataframe with mapping
df_mapping_boro = pd.DataFrame(enc.categories_[0], columns=['arrest_boro'])
df_mapping_boro['boro_ordinal'] = df_mapping_boro.index
df_mapping_boro.head(5)
# save mapping to csv
df_mapping_boro.to_csv('model_dev1/data/processed/mapping_boro.csv', index=False)

############

## Encoding the arrest_precinct column
enc = OrdinalEncoder()
enc.fit(df[['arrest_precinct']])
df['arrest_precinct'] = enc.transform(df[['arrest_precinct']])

# Creating a dataframe with mapping
df_mapping_arrest_precinct = pd.DataFrame(enc.categories_[0], columns=['arrest_precinct'])
df_mapping_arrest_precinct['arrest_precinct_ordinal'] = df_mapping_arrest_precinct.index
df_mapping_arrest_precinct.head(5)
# save mapping to csv
df_mapping_arrest_precinct.to_csv('model_dev1/data/processed/mapping_arrest_precinct.csv', index=False)

############

## Encoding the age_group column
enc = OrdinalEncoder()
enc.fit(df[['age_group']])
df['age_groupp'] = enc.transform(df[['age_group']])

# Creating a dataframe with mapping
df_mapping_age_group = pd.DataFrame(enc.categories_[0], columns=['age_group'])
df_mapping_age_group['age_group_ordinal'] = df_mapping_age_group.index
df_mapping_age_group.head(5)
# save mapping to csv
df_mapping_age_group.to_csv('model_dev1/data/processed/mapping_age_group.csv', index=False)

############

## Encoding the perp_sex column
enc = OrdinalEncoder()
enc.fit(df[['perp_sex']])
df['perp_sex'] = enc.transform(df[['perp_sex']])

# Creating a dataframe with mapping
df_mapping_perp_sex = pd.DataFrame(enc.categories_[0], columns=['perp_sex'])
df_mapping_perp_sex['perp_sex_ordinal'] = df_mapping_perp_sex.index
df_mapping_perp_sex.head(5)
# save mapping to csv
df_mapping_perp_sex.to_csv('model_dev1/data/processed/mapping_perp_sex.csv', index=False)

############

## Encoding the perp_race column
enc = OrdinalEncoder()
enc.fit(df[['perp_race']])
df['perp_race'] = enc.transform(df[['perp_race']])

# Creating a dataframe with mapping
df_mapping_perp_race = pd.DataFrame(enc.categories_[0], columns=['perp_race'])
df_mapping_perp_race['perp_race_ordinal'] = df_mapping_perp_race.index
df_mapping_perp_race.head(5)
# save mapping to csv
df_mapping_perp_race.to_csv('model_dev1/data/processed/mapping_perp_race.csv', index=False)

#######################

## Saving processed dataset to a csv file to test the model
df.to_csv('model_dev1/data/processed/processed_arrest_data.csv', index=False)