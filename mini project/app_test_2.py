# Step 1: Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans


# Step 2: Load Dataset
data = pd.read_csv("data.csv")

print("Dataset Preview:")
print(data.head())


# Step 3: Select Features and Target
X = data[['bedrooms','bathrooms','sqft_living','sqft_lot',
          'floors','condition','sqft_above','sqft_basement','yr_built']]

y = data['price']


# Step 4: Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# ===============================
# 1 Linear Regression
# ===============================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

pred = lr_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))

print("\nLinear Regression RMSE:", rmse)


# ===============================
# 2 Decision Tree
# ===============================
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)

pred = dt_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))

print("Decision Tree RMSE:", rmse)


# ===============================
# 3 Random Forest
# ===============================
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)

pred = rf_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))

print("Random Forest RMSE:", rmse)


# ===============================
# 4 K-Means Clustering
# ===============================
X_cluster = data[['bedrooms','bathrooms','sqft_living','sqft_lot']]

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

data['Cluster'] = kmeans.fit_predict(X_cluster)

print("\nClustered Data Preview:")
print(data.head())