"""
Trains the final tuned Random Forest model (same as Phase 5/6) and saves it,
along with the fitted scaler and expected column structure, to a single file
that the Streamlit app can load instantly without retraining every time.

Run this once before running the app:
    python train_and_save_model.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('customer_churn.csv').drop(columns=['CustomerID'])
X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_prep = X_train.copy()
gender_map = {'Male': 0, 'Female': 1}
X_train_prep['Gender'] = X_train_prep['Gender'].map(gender_map)

categorical_cols = ['Subscription Type', 'Contract Length']
X_train_prep = pd.get_dummies(X_train_prep, columns=categorical_cols, drop_first=True)

numeric_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls',
                'Payment Delay', 'Total Spend', 'Last Interaction']
scaler = StandardScaler()
X_train_prep[numeric_cols] = scaler.fit_transform(X_train_prep[numeric_cols])

# Best parameters found by GridSearchCV in Phase 5
model = RandomForestClassifier(
    n_estimators=200, max_depth=None, min_samples_split=2, random_state=42
)
model.fit(X_train_prep, y_train)

artifact = {
    'model': model,
    'scaler': scaler,
    'reference_columns': list(X_train_prep.columns),
    'numeric_cols': numeric_cols,
}

joblib.dump(artifact, 'churn_model.pkl')
print("Saved churn_model.pkl -- model, scaler, and column structure bundled together.")
