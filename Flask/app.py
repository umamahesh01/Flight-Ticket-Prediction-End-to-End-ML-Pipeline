
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import yaml
import json 
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(BASE_DIR)

# load config relative to project root
config_path = os.path.join(BASE_DIR, "config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)


# Resolve relative artifact paths to absolute
for key in ["data_path", "model_path", "scaler_path", "encoder_path", "feature_path"]:
    config[key] = os.path.join(BASE_DIR, config[key])


# Loading Model
with open(config['model_path'], 'rb') as f:
    model = pickle.load(f)

# Load Encoder
with open(config['encoder_path'], 'rb') as f:
    encoders = pickle.load(f)

# Load Scaler
with open(config['scaler_path'], 'rb') as f:
    scaler = pickle.load(f)

with open(config["feature_path"], "r") as f:
    feature_list = json.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_dict = {
                "airline": request.form.get("airline").strip().title(),   # Spicejet → Spicejet
                "source_city": request.form.get("source_city").strip().title(),
                "departure_time": request.form.get("departure_time").strip().title(),
                "stops": request.form.get("stops").strip().lower(),
                "destination_city": request.form.get("destination_city").strip().title(),
                "class": request.form.get("class_type").strip().title(),
                "days_left": int(request.form.get("days_left"))
            }
        # Keep the same feature order as training
        features_df = pd.DataFrame([input_dict])[feature_list]

        # Encode categorical columns
        for col, le in encoders.items():
            if col in features_df.columns:
                features_df[col] = le.transform(features_df[col].astype(str))

        # Scale features
        features_scaled = scaler.transform(features_df)

        # Predict
        prediction = model.predict(features_scaled)[0]

        return render_template('index.html',
                               prediction_text=f"Estimated Fare: ₹{int(prediction)}")

    except Exception as e:
        return render_template('index.html',
                               prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=True)
