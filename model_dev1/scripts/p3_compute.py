import pandas as pd
import pickle
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Importing the cleaned sample of 10k data
df = pd.read_csv('model_dev1/data/processed/processed_arrest_data.csv')
df = pd.read_csv('model_dev1/data/processed/mapping_perp_sex.csv')

# Dropping rows with missing values
df.dropna(inplace=True)
len(df)

# Defining the features and the target variable ('perp_sex')
X = df.drop('perp_sex', axis=1)  # Features (all columns except 'perp_sex')
y = df['perp_sex']               # Target variable (perp_sex)


# Initializing the StandardScaler
scaler = StandardScaler()
scaler.fit(X)          # Fitting the scaler to the features
pickle.dump(scaler, open('model_dev1/models/scaler.sav', 'wb'))  # Saving the scaler for later use

# Fitting the scaler to the features and transform
X_scaled = scaler.transform(X)

# Splitting the scaled data into training, validation, and testing sets 
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1, random_state=42)

# Checking the size of each set
(X_train.shape, X_val.shape, X_test.shape)

# Pkle the X_train for later use in explanation
pickle.dump(X_train, open('model_dev1/models/X_train.sav', 'wb'))
# Pkle X.columns for later use in explanation
pickle.dump(X.columns, open('model_dev1/models/X_columns.sav', 'wb'))

