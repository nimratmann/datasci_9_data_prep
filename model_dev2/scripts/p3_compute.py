import pandas as pd
import pickle
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# Load the cleaned dataset
df = pd.read_csv('model_dev2/data/processed/processed_hate_crimes.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# Define features and target variable
X = df.drop('offense_description', axis=1)
y = df['offense_description']

# Initialize the StandardScaler
scaler = StandardScaler()
scaler.fit(X)
pickle.dump(scaler, open('model_dev2/models/scaler.sav', 'wb'))

# Transform the features using the scaler
X_scaled = scaler.transform(X)

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Save X_train and feature columns for later use
pickle.dump(X_train, open('model_dev2/models/X_train.sav', 'wb'))
pickle.dump(X.columns, open('model_dev2/models/X_columns.sav', 'wb'))




# Create a baseline model using DummyClassifier
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
dummy_acc = dummy.score(X_val, y_val)

# Create a Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_val_pred = log_reg.predict(X_val)
log_reg_acc = log_reg.score(X_val, y_val)
log_reg_mse = mean_squared_error(y_val, y_val_pred)
log_reg_r2 = r2_score(y_val, y_val_pred)

# Evaluate Logistic Regression model
print(confusion_matrix(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))
print('Baseline accuracy:', dummy_acc)
print('Logistic Regression accuracy:', log_reg_acc)
print('Logistic Regression MSE:', log_reg_mse)
print('Logistic Regression R2:', log_reg_r2)

# Train XGBoost model
xgboost = XGBClassifier()
xgboost.fit(X_train, y_train)
y_val_pred = xgboost.predict(X_val)
xgboost_acc = xgboost.score(X_val, y_val)
xgboost_mse = mean_squared_error(y_val, y_val_pred)
xgboost_r2 = r2_score(y_val, y_val_pred)

# Evaluate XGBoost model
print(confusion_matrix(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))
print('Baseline accuracy:', dummy_acc)
print('Logistic Regression accuracy:', log_reg_acc)
print('Logistic Regression MSE:', log_reg_mse)
print('Logistic Regression R2:', log_reg_r2)
print('XGBoost accuracy:', xgboost_acc)
print('XGBoost MSE:', xgboost_mse)
print('XGBoost R2:', xgboost_r2)

# Perform hyperparameter tuning on XGBoost
param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
}

xgboost = XGBClassifier()
grid_search = GridSearchCV(estimator=xgboost, param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
y_val_pred = grid_search.predict(X_val)
grid_search_acc = grid_search.score(X_val, y_val)
grid_search_mse = mean_squared_error(y_val, y_val_pred)
grid_search_r2 = r2_score(y_val, y_val_pred)

# Evaluate tuned XGBoost model
print(confusion_matrix(y_val, y_val_pred))
print(classification_report(y_val, y_val_pred))
print('Baseline accuracy:', dummy_acc)
print('Logistic Regression accuracy:', log_reg_acc)
print('XGBoost Model 1 accuracy:', xgboost_acc)
print('XGBoost Model 2 accuracy:', grid_search_acc)

# Print the best parameters and the best score
print(grid_search.best_params_)
print(grid_search.best_score_)

# Train a new XGBoost model with the best parameters
xgboost = XGBClassifier(learning_rate=0.1, max_depth=4, n_estimators=100)
xgboost.fit(X_train, y_train)
y_test_pred = xgboost.predict(X_test)
xgboost_acc = xgboost.score(X_test, y_test)

# Save the final XGBoost model 
pickle.dump(xgboost, open('model_dev2/models/xgboost.sav', 'wb'))
