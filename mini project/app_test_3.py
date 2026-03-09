import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


# Load dataset
data = pd.read_csv("data.csv")

print("Dataset Preview:")
print(data.head())


# Features and target
X = data[['bedrooms','bathrooms','sqft_living','sqft_lot',
          'floors','condition','sqft_above','sqft_basement','yr_built']]

y = data['price']


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# Models dictionary
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "SVM": SVR(),
    "KNN": KNeighborsRegressor()
}


# Train and test models
for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))

    print(name, "RMSE:", rmse)