import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# =========================
# Synthetic Data Generation
# =========================

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = pd.DataFrame({
        'Return_ID': np.arange(1, n_samples + 1),
        'Income': np.random.normal(50000, 15000, n_samples),  # Average income of $50,000
        'Deductions': np.random.normal(10000, 5000, n_samples),  # Average deductions of $10,000
        'Dependents': np.random.randint(0, 5, n_samples),  # 0 to 4 dependents
        'Withholding': np.random.normal(8000, 2000, n_samples),  # Taxes withheld
        'Refund_Claimed': np.random.normal(2000, 1000, n_samples),  # Refund claimed
    })

    # Introduce some fraudulent cases
    fraud_indices = np.random.choice(
        data.index, size=int(0.05 * n_samples), replace=False
    )
    data.loc[fraud_indices, 'Fraud'] = 1
    data['Fraud'] = data['Fraud'].fillna(0)

    # Adjust features to simulate fraud
    num_fraud_cases = int(data['Fraud'].sum())

    data.loc[data['Fraud'] == 1, 'Deductions'] *= np.random.uniform(
        2, 5, size=num_fraud_cases
    )
    data.loc[data['Fraud'] == 1, 'Refund_Claimed'] *= np.random.uniform(
        1.5, 3, size=num_fraud_cases
    )

    # Ensure no negative values
    for col in ['Income', 'Deductions', 'Withholding', 'Refund_Claimed']:
        data[col] = data[col].apply(lambda x: abs(x))

    # Reset index to avoid any issues
    data.reset_index(drop=True, inplace=True)
    # Ensure correct data types
    data['Fraud'] = data['Fraud'].astype(int)

    return data

# ================
# Data Preparation
# ================

@st.cache_data
def load_data():
    data = generate_synthetic_data(2000)
    return data

def preprocess_data(df):
    df = df.copy()
    features = ['Income', 'Deductions', 'Dependents', 'Withholding', 'Refund_Claimed']
    X = df[features]
    y = df['Fraud']
    return X, y

# ============
# Build Model
# ============

@st.cache_resource
def train_model(X, y):
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.3, random_state=42
    )

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return model, report, cm

# ===========================
# Streamlit App Starts Here
# ===========================

st.title('Tax Fraud Detection System')

# Load and display data
data = load_data()

st.sidebar.title('Navigation')
options = st.sidebar.radio('Select a page:', ['Data Exploration', 'Model Training', 'Fraud Prediction'])

if options == 'Data Exploration':
    st.header('Data Exploration')
    st.write('This section allows you to explore the dataset.')

    if st.checkbox('Show raw data'):
        st.write(data.head())

    st.subheader('Fraud Class Distribution')
    fig, ax = plt.subplots()
    sns.countplot(x='Fraud', data=data, ax=ax)
    st.pyplot(fig)

    st.subheader('Feature Distributions')
    feature = st.selectbox('Select a feature to visualize:', data.columns[1:-1])
    fig2, ax2 = plt.subplots()
    sns.histplot(data, x=feature, hue='Fraud', kde=True, ax=ax2)
    st.pyplot(fig2)

elif options == 'Model Training':
    st.header('Model Training and Evaluation')
    st.write('In this section, we preprocess the data, handle class imbalance, train the model, and evaluate its performance.')

    X, y = preprocess_data(data)
    model, report, cm = train_model(X, y)

    st.subheader('Classification Report')
    st.text(pd.DataFrame(report).transpose())

    st.subheader('Confusion Matrix')
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    st.pyplot(fig3)

    st.write('The model has been trained and is ready to make predictions.')

elif options == 'Fraud Prediction':
    st.header('Fraud Risk Prediction')
    st.write('Input tax return information to predict the likelihood of fraud.')

    st.subheader('Input Tax Return Data')

    # Input fields
    income = st.number_input('Reported Income', min_value=0.0, value=50000.0)
    deductions = st.number_input('Reported Deductions', min_value=0.0, value=10000.0)
    dependents = st.number_input('Number of Dependents', min_value=0, step=1, value=0)
    withholding = st.number_input('Taxes Withheld', min_value=0.0, value=8000.0)
    refund_claimed = st.number_input('Refund Claimed', min_value=0.0, value=2000.0)

    user_data = pd.DataFrame({
        'Income': [income],
        'Deductions': [deductions],
        'Dependents': [dependents],
        'Withholding': [withholding],
        'Refund_Claimed': [refund_claimed]
    })

    # Load trained model
    X, y = preprocess_data(data)
    model, _, _ = train_model(X, y)

    if st.button('Predict Fraud Risk'):
        prediction = model.predict(user_data)
        prediction_proba = model.predict_proba(user_data)

        if prediction[0] == 1:
            st.error('This tax return is predicted to be **Fraudulent**.')
        else:
            st.success('This tax return is predicted to be **Legitimate**.')

        st.subheader('Prediction Probability')
        st.write(f'Probability of Fraud: {prediction_proba[0][1]:.2%}')
        st.write(f'Probability of Legitimate: {prediction_proba[0][0]:.2%}')

# ============
# Footer
# ============

st.sidebar.info('Developed by [Umair Ali]')
