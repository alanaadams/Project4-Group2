from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved models
lr_model = joblib.load('linear_regression_model1.pkl')
rf_model = joblib.load('random_forest_model1.pkl')
gb_model = joblib.load('gradient_boosting_model1.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the JSON data
    features = [
        float(request.json['sqft_living']),
        float(request.json['sqft_living15']),
        float(request.json['sqft_above']),
        float(request.json['yr_built']),
        float(request.json['sqft_lot15']),
        float(request.json['sqft_lot']),
        float(request.json['grade']),
        float(request.json['id']),
        float(request.json['month_sold']),
        float(request.json['bathrooms']),
        float(request.json['sqft_basement']),
        float(request.json['bedrooms']),
        float(request.json['condition']),
        float(request.json['yr_sold'])
    ]
    
    prediction = rf_model.predict([features])[0]

    return jsonify({'price': prediction})


if __name__ == "__main__":
    app.run(debug=True)
