import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# load dataset
data = pd.read_csv("data.csv")

# features
X = data[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','condition','sqft_above','sqft_basement','yr_built']]

# target
y = data['price']

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# model
model = RandomForestRegressor(n_estimators=100)

# train
model.fit(X_train,y_train)

# predict
pred = model.predict(X_test)

# evaluation
# predict
pred = model.predict(X_test)

# calculate RMSE
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)

print("Model trained successfully")
print("RMSE:", rmse)