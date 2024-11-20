from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np

app = Flask(__name__)

# Load the models
with open('models/credit_card_model.pkl', 'rb') as f:
    credit_model = pickle.load(f)

with open('models/upi_model.pkl', 'rb') as f:
    upi_model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/credit_fraud', methods=['GET', 'POST'])
def credit_fraud():
    if request.method == 'POST':
        features_input = request.form['features']
        try:
            # Process the input (split by comma and convert to float)
            features = [float(x) for x in features_input.split(',')]
            features = np.array(features).reshape(1, -1)

            # Predict fraud using the credit card model
            prediction = credit_model.predict(features)
            if prediction == 1:
                result = "Fraudulent transaction detected"
            else:
                result = "Not a fraudulent transaction"

            return render_template('result.html', title="Credit Fraud Result", result=result)

        except Exception as e:
            return render_template('credit.html', error="Invalid input. Please enter valid features.", features_input=features_input)

    # Render the form
    return render_template('credit.html')

@app.route('/upi_fraud', methods=['GET', 'POST'])
def upi_fraud():
    if request.method == 'POST':
        # Extract form data
        date = request.form['Date']
        category = request.form['Category']
        ref_no = request.form['RefNo']
        date1 = request.form['Date1']
        withdrawal = request.form['Withdrawal']
        deposit = request.form['Deposit']
        balance = request.form['Balance']

        try:
            # Prepare the input data for prediction (as a list or numpy array)
            input_data = [
                float(withdrawal),
                float(deposit),
                float(balance)
                # Add other features based on your model
            ]

            # Predict fraud using the UPI model
            input_data = np.array(input_data).reshape(1, -1)
            prediction = upi_model.predict(input_data)
            if prediction == 1:
                result = "Fraudulent UPI transaction detected"
            else:
                result = "Not a fraudulent UPI transaction"

            return render_template('result.html', title="UPI Fraud Result", result=result)

        except Exception as e:
            return render_template('upi.html', error="Invalid input. Please enter valid data.")

    return render_template('upi.html')

if __name__ == '__main__':
    app.run(debug=True)
