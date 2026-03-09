# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# Step 2: Load dataset
data = pd.read_csv("data.csv")

print("Dataset Preview:")
print(data.head())


# Step 3: Select useful columns
X = data[['bedrooms','bathrooms','sqft_living','sqft_lot',
          'floors','condition','sqft_above','sqft_basement','yr_built']]

y = data['price']


# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Step 5: Create Random Forest model
model = RandomForestRegressor(n_estimators=100)


# Step 6: Train model
model.fit(X_train, y_train)


# Step 7: Make predictions
predictions = model.predict(X_test)


# Step 8: Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))


print("Random Forest RMSE:", rmse)