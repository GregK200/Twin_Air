import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint

# Load and preprocess the data
file_1 = 'device_1_0080E1150510BDE6_2024-12-10_19-00_to_21-00_with_sampling.csv'
file_2 = 'device_1_0080E1150510BDE6_2024-05-10_13-00_to_19-00_with_sampling.csv'
file_3 = 'device_1_0080E1150510BDE6_2024-05-10_18-00_to_21-00_with_sampling.csv'
file_4 = 'device_2_0080E1150533F233_2024-05-10_13-00_to_19-00_with_sampling.csv'
file_5 = 'device_2_0080E1150533F233_2024-05-10_18-00_to_21-00_with_sampling.csv'
file_6 = 'device_2_0080E1150533F233_2024-12-10_19-00_to_21-00_with_sampling.csv'

df_1 = pd.read_csv(file_1)
df_2 = pd.read_csv(file_2)
df_3 = pd.read_csv(file_3)
df_4 = pd.read_csv(file_4)
df_5 = pd.read_csv(file_5)
df_6 = pd.read_csv(file_6)

df_combined = pd.concat([df_1[['Relative_Humidity', 'Temperature', 'MC_PM25', 'NC_PM25']],
                         df_2[['Relative_Humidity', 'Temperature', 'MC_PM25', 'NC_PM25']],
                         df_4[['Relative_Humidity', 'Temperature', 'MC_PM25', 'NC_PM25']],
                         df_5[['Relative_Humidity', 'Temperature', 'MC_PM25', 'NC_PM25']],
                         df_6[['Relative_Humidity', 'Temperature', 'MC_PM25', 'NC_PM25']],
                         df_3[['Relative_Humidity', 'Temperature', 'MC_PM25', 'NC_PM25']]])

df_combined.dropna(inplace=True)
df_combined['future_PM25'] = df_combined['MC_PM25'].shift(-120)
df_combined['future_NC_PM25'] = df_combined['NC_PM25'].shift(-120)  # Shift by 10 minutes
df_combined.dropna(inplace=True)

X = df_combined[['MC_PM25', 'NC_PM25', 'Temperature', 'Relative_Humidity']]
y = df_combined['future_PM25']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning with Randomized Search
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [10, 15, 20, None],
    'min_samples_split': randint(2, 10),
}

rf_model = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist, 
                                   n_iter=100, cv=5, scoring='neg_mean_squared_error', 
                                   verbose=2, n_jobs=1, random_state=42)

# Fit the Randomized Search model
random_search.fit(X_train, y_train)

# Use the best estimator from Randomized Search
rf_model = random_search.best_estimator_

# Train the model and make predictions
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print metrics
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")

# Cross-validation score
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-Validation MSE: {abs(cv_scores.mean()):.2f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual PM2.5')
plt.ylabel('Predicted PM2.5')
plt.title('Predicted vs Actual PM2.5 Levels (Test Set)')
plt.grid()
plt.show()
