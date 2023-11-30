import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder

## Loading raw data (pickle)
df = pd.read_pickle('model_dev2/data/raw/hate_crimes.pkl')

## Getting column names
df.columns

## Cleaning column names 
## Making them all lower case and removing white spaces
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns

## Getting data types
df.dtypes

## Dropping these columns because too missing values
to_drop = [
    'full_complaint_id',
    'complaint_year_number',
    'month_number',
    'complaint_precinct_code',
    'pd_code_description',
    'arrest_date',
    'arrest_id',
    ]
df.drop(to_drop, axis=1, inplace=True, errors='ignore')
df.columns

#######################

## Changing record create date column from object data type to datetime data type
df['record_create_date'] = pd.to_datetime(df['record_create_date'])
# Changing the values to show only the month and year: yyyy-mm for the ordinal encoding
df['record_create_date'] = df['record_create_date'].dt.to_period('M')

# Performing ordinal encoding on date column.
enc = OrdinalEncoder()
enc.fit(df[['record_create_date']])
df['record_create_date'] = enc.transform(df[['record_create_date']])

# Creating a dataframe with mapping
df_mapping_date = pd.DataFrame(enc.categories_[0], columns=['record_create_date'])
df_mapping_date['record_create_date_ordinal'] = df_mapping_date.index
df_mapping_date.head(5)

# Saving mapping to csv
df_mapping_date.to_csv('model_dev2/data/processed/mapping_date.csv', index=False)

############


## Encoding the patrol_borough_name column
enc = OrdinalEncoder()
enc.fit(df[['patrol_borough_name']])
df['patrol_borough_name'] = enc.transform(df[['patrol_borough_name']])

# Creating a dataframe with mapping
df_mapping_borough = pd.DataFrame(enc.categories_[0], columns=['patrol_borough_name'])
df_mapping_borough['borough_ordinal'] = df_mapping_borough.index
df_mapping_borough.head(5)
# save mapping to csv
df_mapping_borough.to_csv('model_dev2/data/processed/mapping_borough.csv', index=False)

############

## Encoding the patrol_borough_name column
enc = OrdinalEncoder()
enc.fit(df[['patrol_borough_name']])
df['patrol_borough_name'] = enc.transform(df[['patrol_borough_name']])

# Creating a dataframe with mapping
df_mapping_borough = pd.DataFrame(enc.categories_[0], columns=['patrol_borough_name'])
df_mapping_borough['borough_ordinal'] = df_mapping_borough.index
df_mapping_borough.head(5)
# save mapping to csv
df_mapping_borough.to_csv('model_dev2/data/processed/mapping_borough.csv', index=False)

############

## Encoding the county column
enc = OrdinalEncoder()
enc.fit(df[['county']])
df['county'] = enc.transform(df[['county']])

# Creating a dataframe with mapping
df_mapping_county = pd.DataFrame(enc.categories_[0], columns=['county'])
df_mapping_county['county_ordinal'] = df_mapping_county.index
df_mapping_county.head(5)
# save mapping to csv
df_mapping_county.to_csv('model_dev2/data/processed/mapping_county.csv', index=False)

############

## Encoding the law_code_category_description column
enc = OrdinalEncoder()
enc.fit(df[['law_code_category_description']])
df['law_code_category_description'] = enc.transform(df[['law_code_category_description']])

# Creating a dataframe with mapping
df_mapping_law_code = pd.DataFrame(enc.categories_[0], columns=['law_code_category_description'])
df_mapping_law_code['law_code_ordinal'] = df_mapping_law_code.index
df_mapping_law_code.head(5)
# save mapping to csv
df_mapping_law_code.to_csv('model_dev2/data/processed/mapping_law_code.csv', index=False)

############

## Encoding the law_code_category_description column
enc = OrdinalEncoder()
enc.fit(df[['law_code_category_description']])
df['law_code_category_description'] = enc.transform(df[['law_code_category_description']])

# Creating a dataframe with mapping
df_mapping_law_code = pd.DataFrame(enc.categories_[0], columns=['law_code_category_description'])
df_mapping_law_code['law_code_ordinal'] = df_mapping_law_code.index
df_mapping_law_code.head(5)
# save mapping to csv
df_mapping_law_code.to_csv('model_dev2/data/processed/mapping_law_code.csv', index=False)

############

## Encoding the offense_description column
enc = OrdinalEncoder()
enc.fit(df[['offense_description']])
df['offense_description'] = enc.transform(df[['offense_description']])

# Creating a dataframe with mapping
df_mapping_offense = pd.DataFrame(enc.categories_[0], columns=['offense_description'])
df_mapping_offense['offense_ordinal'] = df_mapping_offense.index
df_mapping_offense.head(5)
# save mapping to csv
df_mapping_offense.to_csv('model_dev2/data/processed/mapping_offense.csv', index=False)

############

## Encoding the bias_motive_description column
enc = OrdinalEncoder()
enc.fit(df[['bias_motive_description']])
df['bias_motive_description'] = enc.transform(df[['bias_motive_description']])

# Creating a dataframe with mapping
df_mapping_motive = pd.DataFrame(enc.categories_[0], columns=['bias_motive_description'])
df_mapping_motive['motive_ordinal'] = df_mapping_motive.index
df_mapping_motive.head(5)
# save mapping to csv
df_mapping_motive.to_csv('model_dev2/data/processed/mapping_motive.csv', index=False)

############

## Encoding the offense_category column
enc = OrdinalEncoder()
enc.fit(df[['offense_category']])
df['offense_category'] = enc.transform(df[['offense_category']])

# Creating a dataframe with mapping
df_mapping_offense_category = pd.DataFrame(enc.categories_[0], columns=['offense_category'])
df_mapping_offense_category['offense_category_ordinal'] = df_mapping_offense_category.index
df_mapping_offense_category.head(5)
# save mapping to csv
df_mapping_offense_category.to_csv('model_dev2/data/processed/mapping_offense_category.csv', index=False)

############

## Saving processed dataset to a csv file to test the model
df.to_csv('model_dev2/data/processed/processed_hate_crimes.csv', index=False)