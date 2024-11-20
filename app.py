from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import logging
import os
from joblib import load

app = Flask(__name__)

# Logging configuration
logging.basicConfig(level=logging.DEBUG)

# Model loading with error handling
try:
    credit_model = load('models/credit_card_model.pkl')  # You can stick to pickle if preferred
    upi_model = load('models/upi_model.pkl')
except FileNotFoundError:
    raise RuntimeError("Model files are missing. Please ensure all required models are uploaded.")
except Exception as e:
    raise RuntimeError(f"Failed to load models: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/credit_fraud', methods=['GET', 'POST'])
def credit_fraud():
    if request.method == 'POST':
        features_input = request.form['features']
        try:
            features = [float(x) for x in features_input.split(',')]
            features = np.array(features).reshape(1, -1)
            prediction = credit_model.predict(features)
            result = "Fraudulent transaction detected" if prediction == 1 else "Not a fraudulent transaction"
            return render_template('result.html', title="Credit Fraud Result", result=result)
        except Exception as e:
            logging.error(f"Error during credit fraud prediction: {e}")
            return render_template('credit.html', error="Invalid input. Please enter valid features.", features_input=features_input)

    return render_template('credit.html')

@app.route('/upi_fraud', methods=['GET', 'POST'])
def upi_fraud():
    if request.method == 'POST':
        date = request.form['Date']
        category = request.form['Category']
        ref_no = request.form['RefNo']
        date1 = request.form['Date1']
        withdrawal = request.form['Withdrawal']
        deposit = request.form['Deposit']
        balance = request.form['Balance']

        try:
            input_data = [
                float(withdrawal),
                float(deposit),
                float(balance)
            ]
            input_data = np.array(input_data).reshape(1, -1)
            prediction = upi_model.predict(input_data)
            result = "Fraudulent UPI transaction detected" if prediction == 1 else "Not a fraudulent UPI transaction"
            return render_template('result.html', title="UPI Fraud Result", result=result)
        except Exception as e:
            logging.error(f"Error during UPI fraud prediction: {e}")
            return render_template('upi.html', error="Invalid input. Please enter valid data.")

    return render_template('upi.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
