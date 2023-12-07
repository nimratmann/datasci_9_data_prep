# datasci_9_data_prep
Selecting datasets suitable for a machine learning experiment, with an emphasis on data cleaning, encoding, and transformation steps necessary to prepare the data.

# Documentation
## Dataset 1: NYPD Arrest Data (Year to Date)
https://catalog.data.gov/dataset/nypd-arrest-data-year-to-date 

- This dataset provides a detailed overview of every arrest made by the NYPD in New York City throughout the current year. Quarterly, the Office of Management Analysis and Planning manually compiles and examines this data. Each entry provides details about the type of crime, the location and time of the arrest in NYC, and additional information regarding suspect demographics. This dataset serves as a valuable resource for the public to explore the dynamics of police enforcement activities.

- The intended machine learning task for this dataset involves classification, with column 'perp_sex' (perpetrator’s sex description) being the target variable (X).

### Cleaning and Transforming Data

1. Column Name Standardization:

    Initially, the column names underwent standardization by replacing spaces with underscores and converting all characters to lowercase. This ensures consistency and ease of reference.

2. Column Removal:

      Certain columns were identified for removal, as they were deemed unnecessary for the analysis. The script utilized a predefined list (to_drop) containing column names like 'incident_key,' 'occur_time,' and others. These columns were then dropped from the dataset.
```
# Columns to be dropped
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

# Drop specified columns from the DataFrame 'df'
df.drop(to_drop, axis=1, inplace=True, errors='ignore')
```


3. Date Column Transformation:

    The date column was transformed. First, it was converted to a datetime datatype, providing a standardized date format. Subsequently, the days of the week (Monday through Sunday) were extracted from the date and coded into integer values (1 to 7). This enhanced representation facilitates day-wise analysis. A data dictionary detailing these transformations was then saved to a CSV file in the /model_dev1/data/processed folder.

4.  Ordinal Encoding for Multiple Columns:

      Several categorical columns such as 'arrest_boro,' 'arrest_precinct', 'age_group,' 'perp_sex,' and 'perp_race' underwent ordinal encoding. This process assigned integer values to categories, facilitating numerical analysis. Corresponding data dictionaries for these columns were saved in the /model_dev1/data/processed folder.
```
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
```

5. Save Cleaned Data:

   A duplicate of the cleaned and processed dataset was stored in the /model_dev1/data/processed folder. This step ensures that the modified dataset is readily available for subsequent analyses or model development.
   
### Dataset Splitting
In the p3_compute.py file, a script is created to perform the dataset splitting into these three parts. This division is crucial for model development and evaluation:

- Training ```(train_x, train_y)```: The model learns patterns from the training set, adjusting its parameters based on the provided features and labels.
- Validation ```(val_x, val_y)```: During training, the model's performance is regularly evaluated on the validation set to identify issues like overfitting or underfitting and fine-tune hyperparameters accordingly.
- Testing ```(test_x, test_y)```: The final evaluation is conducted on the test set, providing an unbiased assessment of the model's ability to generalize to new, unseen data.





## Dataset 2: NYPD Hate Crimes

https://catalog.data.gov/dataset/nypd-hate-crimes 

- The NYPD Hate Crimes dataset captures incidents of hate crimes reported to the New York Police Department. The dataset covers hate crime incidents in New York, providing a comprehensive view of offenses motivated by bias or prejudice. This dataset enables a comprehensive analysis of hate crimes, allowing for insights into temporal trends, geographic patterns, offense types, and bias motives. The inclusion of arrest-related information adds a layer of understanding law enforcement responses and outcomes. Researchers and analysts can use this dataset to explore the dynamics of hate crimes in New York City over time.

- The intended machine learning task for this dataset involves classification, with column 'offense_description' (a description of the offense) being the target variable (X).

  
### Cleaning and Transforming Data
1. Column Name Standardization:
   
   Standardized column names by converting them to lowercase and replacing spaces with underscores.
```
df.columns = df.columns.str.lower().str.replace(' ', '_')
```

2. Drop Columns with Missing Values:

    Identified and dropped columns with potentially too many missing values.
```
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

```
3. Change Data Types and Ordinal Encoding for Dates:
   
- Converted the 'record_create_date' column to a datetime data type.
- Extracted month and year, then performed ordinal encoding on the date.
```
df['record_create_date'] = pd.to_datetime(df['record_create_date'])
df['record_create_date'] = df['record_create_date'].dt.to_period('M')
enc = OrdinalEncoder()
enc.fit(df[['record_create_date']])
df['record_create_date'] = enc.transform(df[['record_create_date']])
```       
4. Ordinal Encoding for Categorical Columns:

    Applied ordinal encoding to several categorical columns, including 'patrol_borough_name,' 'county,' 'law_code_category_description,' 'offense_description,' 'bias_motive_description,' and 'offense_category.'
```
# Example for 'patrol_borough_name'
enc = OrdinalEncoder()
enc.fit(df[['patrol_borough_name']])
df['patrol_borough_name'] = enc.transform(df[['patrol_borough_name']])
# Similar encoding for other categorical columns
```

5. Create DataFrames with Mapping Information:

   For each ordinal encoding process, created a dataframe with mapping information to capture the original categorical values and their corresponding encoded numerical values.
```
# Example for 'patrol_borough_name'
df_mapping_borough = pd.DataFrame(enc.categories_[0], columns=['patrol_borough_name'])
df_mapping_borough['borough_ordinal'] = df_mapping_borough.index
df_mapping_borough.to_csv('model_dev2/data/processed/mapping_borough.csv', index=False)
# Similar dataframes for other ordinal encoding processes
```
6. Save Processed Dataset to CSV:

    Saved the final processed dataset to a CSV file for testing models.

```
df.to_csv('model_dev2/data/processed/processed_hate_crimes.csv', index=False)
```

### Dataset Splitting
