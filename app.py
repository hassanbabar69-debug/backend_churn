import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
info = joblib.load(os.path.join(BASE_DIR, 'features.pkl'))

FEATURE_COLUMNS = info['feature_columns']
NUMERIC_FEATURES = info['numeric_features']
BINARY_MAPS = info['binary_maps']
MULTI_COLS = info['multi_cols']

CHOICES = {
    'gender': ['Female', 'Male'],
    'SeniorCitizen': ['0', '1'],
    'Partner': ['Yes', 'No'],
    'Dependents': ['Yes', 'No'],
    'PhoneService': ['Yes', 'No'],
    'MultipleLines': ['Yes', 'No'],
    'InternetService': ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity': ['Yes', 'No'],
    'OnlineBackup': ['Yes', 'No'],
    'DeviceProtection': ['Yes', 'No'],
    'TechSupport': ['Yes', 'No'],
    'StreamingTV': ['Yes', 'No'],
    'StreamingMovies': ['Yes', 'No'],
    'Contract': ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod': ['Electronic check', 'Mailed check',
                      'Bank transfer (automatic)', 'Credit card (automatic)'],
}
LABELS = {
    'gender': 'Gender', 'SeniorCitizen': 'Senior Citizen', 'Partner': 'Has Partner',
    'Dependents': 'Has Dependents', 'PhoneService': 'Phone Service',
    'MultipleLines': 'Multiple Lines', 'InternetService': 'Internet Service',
    'OnlineSecurity': 'Online Security', 'OnlineBackup': 'Online Backup',
    'DeviceProtection': 'Device Protection', 'TechSupport': 'Tech Support',
    'StreamingTV': 'Streaming TV', 'StreamingMovies': 'Streaming Movies',
    'Contract': 'Contract Type', 'PaperlessBilling': 'Paperless Billing',
    'PaymentMethod': 'Payment Method',
}
NUMERIC_RANGES = {            
    'tenure': (0, 100, 'Tenure (months)'),
    'MonthlyCharges': (0, 500, 'Monthly charges'),
    'TotalCharges': (0, 20000, 'Total charges'),
}


def validate(form):
    """Check every field. Returns (clean_values, list_of_error_messages)."""
    errors, clean = [], {}
    for field, options in CHOICES.items():
        value = form.get(field, '').strip()
        if value == '':
            errors.append(f"Please select an option for {LABELS[field]}.")
        elif value not in options:
            errors.append(f"{LABELS[field]} has an invalid value. Please choose from the list.")
        else:
            clean[field] = value
    for field, (low, high, label) in NUMERIC_RANGES.items():
        raw = form.get(field, '').strip()
        if raw == '':
            errors.append(f"Please enter {label}.")
            continue
        try:
            number = float(raw)
        except ValueError:
            errors.append(f"{label} must be a number.")
            continue
        if number < low or number > high:
            errors.append(f"{label} must be between {low} and {high}.")
        else:
            clean[field] = number
    return clean, errors


def preprocess(values):
    """Apply the SAME preprocessing used during training."""
    row = {col: 0 for col in FEATURE_COLUMNS}      

    row['SeniorCitizen'] = int(values['SeniorCitizen'])
    for col, mapping in BINARY_MAPS.items():        
        row[col] = mapping[values[col]]
    for col in MULTI_COLS:                        
        dummy = f"{col}_{values[col]}"
        if dummy in row:                          
            row[dummy] = 1
    for col in NUMERIC_FEATURES:
        row[col] = values[col]

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)   
    df[NUMERIC_FEATURES] = scaler.transform(df[NUMERIC_FEATURES]) 
    return df


def predict_churn(values):
    X = preprocess(values)
    label = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])
    return label, probability


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', choices=CHOICES, form={}, errors=[], result=None)


@app.route('/predict', methods=['POST'])
def predict():
    values, errors = validate(request.form)
    if errors:
        return render_template('index.html', choices=CHOICES, form=request.form,
                               errors=errors, result=None)
    label, probability = predict_churn(values)
    result = {
        'churn': label == 1,
        'percent': round(probability * 100, 1),
    }
    return render_template('index.html', choices=CHOICES, form=request.form,
                           errors=[], result=result)


if __name__ == '__main__':
    app.run(debug=False)
