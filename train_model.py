import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


df = pd.read_excel('WA_Fn-UseC_-Telco-Customer-Churn.xlsx')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df = df.drop(columns=['customerID']).drop_duplicates()

service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in service_cols:
    df[col] = df[col].replace('No internet service', 'No')
df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')


df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
               'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
multi_cols = ['InternetService', 'Contract', 'PaymentMethod']

binary_maps = {'gender': {'Female': 0, 'Male': 1}}
for col in binary_cols:
    if col != 'gender':
        binary_maps[col] = {'No': 0, 'Yes': 1}
    df[col] = df[col].map(binary_maps[col])

df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
X = df.drop(columns=['Churn'])
y = df['Churn']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()
X_train_s = X_train.copy()
X_test_s = X_test.copy()
X_train_s[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_s[numeric_features] = scaler.transform(X_test[numeric_features])


model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_s, y_train)
print("Test accuracy:", round(model.score(X_test_s, y_test), 4))


joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump({'feature_columns': X.columns.tolist(),
             'numeric_features': numeric_features,
             'binary_maps': binary_maps,
             'multi_cols': multi_cols,
             'service_cols': service_cols}, 'features.pkl')
print("Saved: model.pkl, scaler.pkl, features.pkl")
print("Feature order:", X.columns.tolist())
