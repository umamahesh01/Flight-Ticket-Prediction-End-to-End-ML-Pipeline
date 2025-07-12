# ✈️ Flight Price Prediction System

A real-time, production-level Machine Learning project that predicts airline ticket prices using features like airline, source city, destination city, class, stops, and days left to departure.  
Built with a modular, scalable architecture and deployed using a Flask web application.

---

## 🚀 Project Highlights

- ✅ **End-to-End ML pipeline**: Data → Preprocessing → Modeling → Evaluation → Deployment  
- 🧱 **Modular structure** following real-time production codebase standards  
- 🧠 **Machine Learning** using Linear Regression  
- 🧼 **Custom strategies** for Missing Value Handling & Label Encoding  
- 🌐 **Flask-based web interface** for real-time prediction  
- 📦 **Model persistence** using Pickle  
- 💡 **Scalable** for advanced models (Random Forest, XGBoost, etc.)

---

## 📁 Project Structure

```
Flight_Fare_Prediction/
├── data/                         # Raw or cleaned datasets
├── artifacts/                   # Trained model (model.pkl)
├── src/                         # Core ML pipeline
│   ├── base/                    # Base strategy classes
│   ├── data/                    # Data loading logic
│   ├── preprocessing/
│   │   ├── feature_engineering/ # Encoding, scaling
│   │   └── missing_value_handler.py
│   ├── model/                   # Training & evaluation
├── templates/                   # HTML templates for Flask
├── static/                      # CSS for UI
├── main.py                      # Model pipeline entry point
├── app.py                       # Flask application
├── config.yaml                  # Config file for paths/features
└── requirements.txt             # Project dependencies
```


---

## ⚙️ How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/umamahesh01/flight-price-prediction-system.git
   cd flight-price-prediction-system
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model**
   ```bash
   python main.py
   ```

4. **Launch Flask App**
   ```bash
   python app.py
   ```

---

## 🌐 Web Application

- Fill in flight details using the web interface  
- Click **Predict** to get the estimated fare  
- Responsive and intuitive UI for quick use  

---

## 🧠 Future Enhancements

- 🌍 Deploy on **AWS / GCP / Render**
- 🐳 **Dockerize** the entire app
- 🌲 Use **Random Forest / XGBoost** for better accuracy
- 📊 Integrate **MLflow** or **Weights & Biases** for experiment tracking
- 🔄 Add **CI/CD pipelines** with GitHub Actions

---

## 👨‍💻 Author

**Uma Mahesh Reddy**  
[GitHub – @umamahesh01](https://github.com/umamahesh01)

---

## 📄 License

This project is licensed under the **MIT License**.
