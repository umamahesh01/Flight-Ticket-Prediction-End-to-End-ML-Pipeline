✈️ Flight Price Prediction System
A real-time, production-level Machine Learning project that predicts airline ticket prices using features like airline, source city, destination city, class, stops, and days left to departure. Built with a modular, scalable architecture, and served via a Flask web application.

🚀 Project Highlights
✅ End-to-End ML pipeline: Data → Preprocessing → Modeling → Evaluation → Deployment

🧱 Modular structure (real-time codebase standards)

🧠 Machine Learning with Linear Regression

🧼 Custom Missing Value & Label Encoding Strategies

🌐 Flask-based web interface

📦 Model persistence with Pickle

💡 Scalable for advanced models (XGBoost, RF, etc.)

📁 Project Architecture
graphql
Copy
Edit
Flight_Fare_Prediction/
│
├── data/                           # Raw or preprocessed datasets
├── artifacts/                      # Trained model (.pkl) saved here
├── src/
│   ├── base/                       # Base classes for Strategy Pattern
│   ├── data/                       # Data loading modules
│   ├── preprocessing/
│   │   ├── feature_engineering/    # Label encoding, scaling
│   │   └── missing_value_handler.py
│   ├── model/                      # Training and evaluation logic
├── templates/                      # HTML templates for Flask app
├── static/                         # CSS and static files
├── main.py                         # Model pipeline script
├── app.py                          # Flask web app
├── config.yaml                     # YAML config (paths, columns)
└── requirements.txt
📊 Dataset Features
airline

flight

source_city

departure_time

stops

arrival_time

destination_city

class

duration

days_left

price (Target)

📈 Model Performance
Algorithm: Linear Regression

R² Score: ~0.90

RMSE: ~7013

MSE: ~49M

🔥 Accurate enough for baseline predictions. Easily extendable to tree-based models for higher accuracy.

⚙️ How to Run
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/umamahesh01/flight-price-prediction-system.git
cd flight-price-prediction-system
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Train the model
bash
Copy
Edit
python main.py
4. Launch Flask App
bash
Copy
Edit
python app.py
🌐 Web Application
Enter flight details on a clean UI.

Click Predict to see estimated fare.

Fast, responsive, and intuitive.

(optional)

🧠 Future Enhancements
Deploy on AWS/GCP/Render

Dockerize app for production

Use advanced models (Random Forest, XGBoost)

Track experiments with MLflow or Weights & Biases

Add CI/CD with GitHub Actions

