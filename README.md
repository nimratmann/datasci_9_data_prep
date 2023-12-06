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

4. Date Column Transformation:

    The date column was transformed. First, it was converted to a datetime datatype, providing a standardized date format. Subsequently, the days of the week (Monday through Sunday) were extracted from the date and coded into integer values (1 to 7). This enhanced representation facilitates day-wise analysis. A data dictionary detailing these transformations was then saved to a CSV file in the /model_dev1/data/processed folder.

5. Ordinal Encoding for Multiple Columns:

      Several categorical columns such as 'boro,' 'statistical_murder_flag,' 'perp_age_group,' 'perp_sex,' 'perp_race,' 'vic_age_group,' 'vic_sex,' and 'vic_race' underwent ordinal encoding. This process assigned integer values to categories, facilitating numerical analysis. Corresponding data dictionaries for these columns were saved in the /model_dev1/data/processed folder.

6. Save Cleaned Data:

   A duplicate of the cleaned and processed dataset was stored in the /model_dev1/data/processed folder. This step ensures that the modified dataset is readily available for subsequent analyses or model development.

## Dataset 2

### Cleaning and Transforming Data
