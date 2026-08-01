import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load model, scaler, and feature names
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'feature_names.pkl')

model = None
scaler = None
feature_names = None

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print("Model, scaler, and features successfully loaded.")
except Exception as e:
    print(f"Error loading model artifacts: {e}")

@app.route('/')
def home():
    """Serves the interactive web user interface."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts patient details as JSON or form data, 
    preprocesses inputs, and returns the prediction as JSON.
    """
    if model is None or scaler is None or feature_names is None:
        return jsonify({"error": "Model files are not loaded correctly on the server."}), 500

    try:
        # Determine content type and extract parameters
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        # Parse and order features according to the model's training layout
        input_features = []
        for feature in feature_names:
            if feature not in data:
                return jsonify({"error": f"Missing required feature: '{feature}'"}), 400
            
            # Cast feature value to numeric
            val = float(data[feature])
            input_features.append(val)

        # Convert to DataFrame with feature names and scale the features
        features_df = pd.DataFrame([input_features], columns=feature_names)
        scaled_features = scaler.transform(features_df)

        # Generate prediction
        prediction = model.predict(scaled_features)[0]
        
        # In this dataset: 1 = No Heart Disease (Normal), 2 = Heart Disease Detected (Abnormal)
        prediction_text = "Heart Disease Detected" if prediction == 2 else "No Heart Disease Detected"

        return jsonify({
            "prediction": prediction_text,
            "raw_prediction": int(prediction)
        })

    except ValueError as ve:
        return jsonify({"error": f"Invalid parameter value type. All inputs must be numeric. Details: {ve}"}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    # Bind to PORT if provided by environment (needed for Render deployment), default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
