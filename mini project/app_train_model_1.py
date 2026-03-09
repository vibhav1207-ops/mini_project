# Import libraries
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Load dataset
data = pd.read_csv("data.csv")

print("Dataset Preview:")
print(data.head())


# Features and target
X = data[['bedrooms','bathrooms','sqft_living','sqft_lot',
          'floors','condition','sqft_above','sqft_basement','yr_built']]

y = data['price']


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)


# Predict
pred = model.predict(X_test)


# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("Random Forest RMSE:", rmse)


# Save model as pkl
joblib.dump(model, "house_price_model.pkl")

print("Model saved as house_price_model.pkl")