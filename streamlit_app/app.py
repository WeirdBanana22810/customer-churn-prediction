"""
Customer Churn Prediction System -- Streamlit App

Loads the model trained by train_and_save_model.py and provides a form-based
UI: enter a customer's details, get a prediction, probability, risk level,
contributing factors, and recommended action. Same logic as the
generate_customer_report() function built in Phase 6 -- just with a visual
front end instead of printed text.
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉", layout="centered")


@st.cache_resource
def load_artifacts():
    return joblib.load('churn_model.pkl')


artifact = load_artifacts()
model = artifact['model']
scaler = artifact['scaler']
reference_columns = artifact['reference_columns']
numeric_cols = artifact['numeric_cols']


# ---------- Same logic as Phase 6's prediction system ----------

def preprocess_customer(customer):
    row = pd.DataFrame([customer])
    row['Gender'] = row['Gender'].map({'Male': 0, 'Female': 1})
    row = pd.get_dummies(row, columns=['Subscription Type', 'Contract Length'], drop_first=True)
    row = row.reindex(columns=reference_columns, fill_value=0)
    row[numeric_cols] = scaler.transform(row[numeric_cols])
    return row


def categorize_risk(probability):
    if probability >= 0.7:
        return 'High'
    elif probability >= 0.3:
        return 'Medium'
    else:
        return 'Low'


def identify_risk_factors(customer):
    factors = []
    if customer['Payment Delay'] > 15:
        factors.append(f"Payment delay of {customer['Payment Delay']} days (past the 15-day danger threshold)")
    if customer['Support Calls'] >= 5:
        factors.append(f"{customer['Support Calls']} support calls (at/above the 5-call danger threshold)")
    if customer['Contract Length'] == 'Monthly':
        factors.append("On a Monthly contract (higher churn rate than Quarterly/Annual)")
    if customer['Gender'] == 'Female':
        factors.append("Gender shows a higher churn rate in this dataset (secondary factor)")
    if not factors:
        factors.append("No major risk factors detected -- customer profile looks stable")
    return factors


def recommend_actions(risk_level, factors):
    actions = []
    if risk_level == 'Low':
        actions.append("No action needed -- maintain current experience.")
        return actions

    factor_text = ' '.join(factors)
    if 'Payment delay' in factor_text:
        actions.append("Resolve billing friction: reach out about the payment delay, "
                        "offer a flexible payment plan or reminder system.")
    if 'support calls' in factor_text.lower():
        actions.append("Escalate to a retention specialist: repeated support calls "
                        "signal unresolved frustration -- proactive outreach needed.")
    if 'Monthly contract' in factor_text:
        actions.append("Offer an incentive to switch to a Quarterly or Annual plan "
                        "to increase commitment and lower churn risk.")

    if risk_level == 'High' and not actions:
        actions.append("Model flags high risk without a clear single driver -- "
                        "manually review this customer's full profile.")
    elif risk_level == 'Medium':
        actions.append("Monitor this customer and consider a light engagement nudge "
                        "(check-in email, usage tips) before risk escalates.")

    return actions


# ---------- UI ----------

st.title("📉 Customer Churn Prediction System")
st.write("Enter a customer's details to predict their churn risk, powered by the "
         "tuned Random Forest model built in this project (~99.8% F1-score on test data).")

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        usage_frequency = st.number_input("Usage Frequency (times/month)", min_value=0, max_value=50, value=15)
        support_calls = st.number_input("Support Calls", min_value=0, max_value=20, value=2)

    with col2:
        payment_delay = st.number_input("Payment Delay (days)", min_value=0, max_value=60, value=5)
        subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
        contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
        total_spend = st.number_input("Total Spend", min_value=0, max_value=5000, value=500)
        last_interaction = st.number_input("Last Interaction (days ago)", min_value=0, max_value=60, value=10)

    submitted = st.form_submit_button("Check Churn Risk", use_container_width=True)

if submitted:
    customer = {
        'Age': age, 'Gender': gender, 'Tenure': tenure, 'Usage Frequency': usage_frequency,
        'Support Calls': support_calls, 'Payment Delay': payment_delay,
        'Subscription Type': subscription_type, 'Contract Length': contract_length,
        'Total Spend': total_spend, 'Last Interaction': last_interaction,
    }

    row = preprocess_customer(customer)
    prediction = model.predict(row)[0]
    probability = model.predict_proba(row)[0][1]
    risk_level = categorize_risk(probability)
    factors = identify_risk_factors(customer)
    actions = recommend_actions(risk_level, factors)

    st.divider()
    st.subheader("Result")

    if prediction == 1:
        st.error("⚠️ Likely to Churn")
    else:
        st.success("✅ Likely to Stay")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Churn Probability", f"{probability:.1%}")
    with col_b:
        risk_display = {"Low": "🟢 Low", "Medium": "🟡 Medium", "High": "🔴 High"}
        st.metric("Risk Level", risk_display[risk_level])

    st.progress(min(float(probability), 1.0))

    st.markdown("**Contributing Factors:**")
    for f in factors:
        st.markdown(f"- {f}")

    st.markdown("**Recommended Actions:**")
    for a in actions:
        st.markdown(f"- {a}")

st.divider()
st.caption("Built as part of a 7-phase Customer Churn Prediction System project. "
           "Model: Random Forest (tuned via GridSearchCV) | Dataset: 64,374 customers.")
