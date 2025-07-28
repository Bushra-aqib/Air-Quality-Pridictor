# model.py
# Description: This script loads air quality data, trains a multi-output regression model, evaluates it,
# and saves the trained model and related files for prediction use.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump

# === Step 1: Load and Clean Data ===
# Load the dataset using proper separator and decimal format
file_path = r"C:\Users\FIREFLY LAPTOPS\Desktop\bushra aqib\air quality app\data\air_quality.csv"
df = pd.read_csv(file_path, sep=';', decimal=',')

# Drop columns and rows with all NaN values
df = df.dropna(axis=1, how='all').dropna()

# === Step 2: Select Features and Targets ===
# Input features (sensor readings and environment factors)
features = ['T', 'RH', 'AH', 'PT08.S1(CO)', 'PT08.S5(O3)']
X = df[features]

# Output targets (pollutants to predict)
targets = ['CO(GT)', 'NOx(GT)', 'NO2(GT)', 'C6H6(GT)']
y = df[targets]

# === Step 3: Split the Data ===
# Use 80% data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Step 4: Train the Model ===
# Create a Linear Regression model for each output target
base_model = LinearRegression()
model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

# === Step 5: Evaluate the Model ===
y_pred = model.predict(X_test)
print("Evaluation for each target:")
for i, target in enumerate(targets):
    mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"{target}: MSE = {mse:.2f}, R² = {r2:.2f}")

# === Step 6: Save the Model and Related Files ===
# Save the trained model
dump(model, "air_quality_multi_model.pkl")

# Save the feature names used during training
dump(features, "features.pkl")

# Save test data for optional evaluation or future comparison
dump(X_test, "X_test.pkl")
dump(y_test, "y_test.pkl")

print("Model and data saved successfully.")


