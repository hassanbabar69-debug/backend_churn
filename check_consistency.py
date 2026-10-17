import numpy as np
import pandas as pd
from app import preprocess, model, scaler, info


df = pd.read_excel('WA_Fn-UseC_-Telco-Customer-Churn.xlsx')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df = df.drop(columns=['customerID']).drop_duplicates().reset_index(drop=True)
raw = df.copy()                                    

for c in info['service_cols']:
    df[c] = df[c].replace('No internet service', 'No')
df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
for col, mapping in info['binary_maps'].items():
    df[col] = df[col].map(mapping)
df = pd.get_dummies(df, columns=info['multi_cols'], drop_first=True)
X = df.drop(columns=['Churn'])[info['feature_columns']]
X_scaled = X.copy()
X_scaled[info['numeric_features']] = scaler.transform(X[info['numeric_features']])

same_features, same_pred = 0, 0
idx = raw.sample(500, random_state=3).index
for i in idx:
    r = raw.loc[i]
    values = {c: str(r[c]) for c in raw.columns if c not in ('Churn', 'tenure', 'MonthlyCharges', 'TotalCharges')}
    for c in info['service_cols']:
        if values[c] == 'No internet service': values[c] = 'No'
    if values['MultipleLines'] == 'No phone service': values['MultipleLines'] = 'No'
    for c in ('tenure', 'MonthlyCharges', 'TotalCharges'):
        values[c] = float(r[c])
    web_row = preprocess(values)
    same_features += np.allclose(web_row.values.astype(float), X_scaled.loc[[i]].values.astype(float))
    same_pred += int(model.predict(web_row)[0]) == int(model.predict(X_scaled.loc[[i]])[0])

print(f"Website feature vector == training feature vector: {same_features}/500 customers")
print(f"Website prediction   == model prediction:          {same_pred}/500 customers")
