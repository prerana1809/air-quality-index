import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model = joblib.load("aqi_lr.pkl")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    predicted_aqi = None
    if request.method == 'POST':
        try:
            # Get user input
            precipitation = float(request.form['precipitation'])
            temp = float(request.form['temp'])
            wind = float(request.form['wind'])

            # Prepare input array
            input_features = np.array([[precipitation, wind, temp]])

            # Predict AQI
            predicted_aqi = model.predict(input_features)[0]
        except Exception as e:
            predicted_aqi = f"Error: {str(e)}"

    return render_template('predict.html', predicted_aqi=predicted_aqi)

@app.route('/notebook')
def notebook():
    return render_template('notebook.html')

if __name__ == '__main__':
    app.run(debug=True)
