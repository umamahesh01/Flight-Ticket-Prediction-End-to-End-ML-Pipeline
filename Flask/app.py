from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your model (make sure model.pkl exists in artifacts/)
model = pickle.load(open('C:\\Users\\mahesh\\OneDrive\\Documents\\Naresh_IT\\LIGHT\\SPEED\\Data_science_RESTART\\CodeBase_Production\\Flight_Fare_prediction\\artifacts\\model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            request.form.get('airline'),
            request.form.get('flight'),
            request.form.get('source_city'),
            request.form.get('departure_time'),
            request.form.get('stops'),
            request.form.get('arrival_time'),
            request.form.get('destination_city'),
            request.form.get('class_type'),
            float(request.form.get('duration')),
            int(request.form.get('days_left'))
        ]

        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)[0]

        return render_template('index.html', prediction_text=f"Estimated Fare: ₹{int(prediction)}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
