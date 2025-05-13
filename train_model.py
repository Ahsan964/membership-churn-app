import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib

# Step 1: Load the dataset
df = pd.read_csv('cleaned_membership_data.csv')

# Step 2: Convert date columns to datetime and create new feature
df['START_DATE'] = pd.to_datetime(df['START_DATE'])
df['END_DATE'] = pd.to_datetime(df['END_DATE'])
df['MEMBERSHIP_DURATION_DAYS'] = (df['END_DATE'] - df['START_DATE']).dt.days

# Step 3: Encode categorical columns
categorical_cols = ['MEMBER_GENDER', 'MEMBER_MARITAL_STATUS', 'MEMBERSHIP_PACKAGE', 'PAYMENT_MODE']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 4: Encode the target variable
df['MEMBERSHIP_STATUS'] = df['MEMBERSHIP_STATUS'].map({'CANCELLED': 0, 'INFORCE': 1})

# Step 5: Select features and target
features = [
    'MEMBERSHIP_DURATION_DAYS',
    'MEMBER_GENDER',
    'MEMBER_MARITAL_STATUS',
    'MEMBER_AGE_AT_ISSUE',
    'MEMBER_OCCUPATION_CD',
    'MEMBER_ANNUAL_INCOME',
    'MEMBERSHIP_PACKAGE',
    'ANNUAL_FEES',
    'ADDITIONAL_MEMBERS',
    'PAYMENT_MODE'
]

X = df[features]
y = df['MEMBERSHIP_STATUS']

# Step 6: Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Train model
knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
knn_model.fit(X_train_scaled, y_train)

# Step 8: Save model and scaler
joblib.dump(knn_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model and scaler saved successfully.")
