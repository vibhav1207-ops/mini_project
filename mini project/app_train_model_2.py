from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load trained model
model = joblib.load("house_price_model.pkl")

# Create Flask app
app = Flask(__name__)

# Home route
@app.route("/docs")
def home():
    return "House Price Prediction API Running"


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    bedrooms = data["bedrooms"]
    bathrooms = data["bathrooms"]
    sqft_living = data["sqft_living"]
    sqft_lot = data["sqft_lot"]
    floors = data["floors"]
    condition = data["condition"]
    sqft_above = data["sqft_above"]
    sqft_basement = data["sqft_basement"]
    yr_built = data["yr_built"]

    features = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot,
                          floors, condition, sqft_above, sqft_basement, yr_built]])

    prediction = model.predict(features)

    return jsonify({
        "predicted_price": float(prediction[0])
    })


# Run server
if __name__ == "__main__":
    app.run(debug=True)