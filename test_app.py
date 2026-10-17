import pandas as pd
import joblib
from app import app, preprocess, predict_churn, model, scaler, info


raw = pd.read_excel('WA_Fn-UseC_-Telco-Customer-Churn.xlsx')
raw['TotalCharges'] = pd.to_numeric(raw['TotalCharges'], errors='coerce').fillna(0)

sample = raw.sample(300, random_state=7)
matches = 0
for _, r in sample.iterrows():
    values = {c: str(r[c]) for c in raw.columns if c not in ('customerID', 'Churn', 'tenure', 'MonthlyCharges', 'TotalCharges')}
    for c in info['service_cols']:
        if values[c] == 'No internet service': values[c] = 'No'
    if values['MultipleLines'] == 'No phone service': values['MultipleLines'] = 'No'
    values['tenure'] = float(r['tenure']); values['MonthlyCharges'] = float(r['MonthlyCharges']); values['TotalCharges'] = float(r['TotalCharges'])
    web_label, _ = predict_churn(values)
    # reference: same row through model directly using the training-style pipeline
    ref = preprocess(values)
    ref_label = int(model.predict(ref)[0])
    matches += (web_label == ref_label)
print(f"Test 1 - web vs model consistency on 300 real rows: {matches}/300")


client = app.test_client()

churn_case = {
    'gender': 'Female', 'SeniorCitizen': '0', 'Partner': 'No', 'Dependents': 'No',
    'tenure': '2', 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
    'StreamingTV': 'Yes', 'StreamingMovies': 'Yes', 'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes', 'PaymentMethod': 'Electronic check',
    'MonthlyCharges': '95.5', 'TotalCharges': '191'}

no_churn_case = {
    'gender': 'Male', 'SeniorCitizen': '0', 'Partner': 'Yes', 'Dependents': 'Yes',
    'tenure': '60', 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'DSL',
    'OnlineSecurity': 'Yes', 'OnlineBackup': 'Yes', 'DeviceProtection': 'Yes', 'TechSupport': 'Yes',
    'StreamingTV': 'No', 'StreamingMovies': 'No', 'Contract': 'Two year',
    'PaperlessBilling': 'No', 'PaymentMethod': 'Credit card (automatic)',
    'MonthlyCharges': '65', 'TotalCharges': '3900'}

def show(name, resp):
    html = resp.get_data(as_text=True)
    if 'Prediction: Churn' in html: out = 'Prediction: Churn'
    elif 'Prediction: No Churn' in html: out = 'Prediction: No Churn'
    elif 'error-box' in html: out = 'Validation errors shown'
    else: out = '??'
    import re
    pct = re.search(r'<strong>([\d.]+)%</strong>', html)
    errs = re.findall(r'<li>(.*?)</li>', html)
    print(f"{name:32s} status={resp.status_code}  ->  {out}" + (f" ({pct.group(1)}%)" if pct else ""))
    for e in errs: print("      -", e)

print("\nTest 2 - web form submissions:")
print("GET / status:", client.get('/').status_code)
show("Expected CHURN customer", client.post('/predict', data=churn_case))
show("Expected NO-CHURN customer", client.post('/predict', data=no_churn_case))
show("Empty form", client.post('/predict', data={}))

bad = dict(churn_case); bad['tenure'] = 'abc'; bad['MonthlyCharges'] = '-5'
show("Text/negative numbers", client.post('/predict', data=bad))

bad2 = dict(churn_case); bad2['Contract'] = 'Lifetime'; bad2['tenure'] = ''
show("Invalid dropdown + empty field", client.post('/predict', data=bad2))
