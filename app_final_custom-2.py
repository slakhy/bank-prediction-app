import streamlit as st
import pickle
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Function to load the trained model
def load_model(model_path):
    with open(model_path, "rb") as file:
        loaded_object = pickle.load(file)
    if isinstance(loaded_object, dict):
        return loaded_object.get("model", None)
    return loaded_object

# Sidebar for model selection
st.sidebar.title("Select Model")
models_available = {
    "Model 1: Decision Tree": "best_model_1_decision_tree.pkl",
    "Model 2: Logistic Regression (Robust Scaler)": "best_model_2_log_reg_RobustScaler.pkl",
    "Model 3: Logistic Regression (Standard Scaler)": "best_model_3_log_reg_StandardScaler.pkl",
}
selected_model_name = st.sidebar.selectbox("Choose a model", list(models_available.keys()))

# Display model information
model_info = {
    "Model 1: Decision Tree": """
    ### Model Information: Model 1
    - **Type**: Decision Tree Classifier
    - **Details**: 
      - Max Depth: 5
      - Min Samples Split: 2
    - **Accuracy**: 0.9108 (Cross-validated)
    - **F1 Score**: 0.9082
    - **Precision**: 0.9120
    - **Recall**: 0.9053
    """,
    "Model 2: Logistic Regression (Robust Scaler)": """
    ### Model Information: Model 2
    - **Type**: Logistic Regression
    - **Scaler**: Robust Scaler
    - **Details**: 
      - C: 10
      - Penalty: l2
      - Solver: lbfgs
    - **Accuracy**: 0.9162 (Cross-validated)
    - **F1 Score**: 0.9007
    - **Precision**: 0.8980
    - **Recall**: 0.9090
    """,
    "Model 3: Logistic Regression (Standard Scaler)": """
    ### Model Information: Model 3
    - **Type**: Logistic Regression
    - **Scaler**: Standard Scaler
    - **Details**: 
      - C: 10
      - Penalty: l2
      - Solver: lbfgs
    - **Accuracy**: 0.9175 (Cross-validated)
    - **F1 Score**: 0.8997
    - **Precision**: 0.8968
    - **Recall**: 0.9078
    """
}
st.sidebar.markdown(model_info[selected_model_name])

# Load the selected model
model_path = models_available[selected_model_name]
model = load_model(model_path)

# Page Title
st.title("ADA 442 Statistical Learning | Classification")
st.markdown("""
### Final Project Assignment: Bank Marketing Data Classification
**Objective**  
Predict whether a client will subscribe to a term deposit using a trained ML model based on direct marketing data from a Portuguese bank.
""")

# User Input Section
st.header("Enter Prediction Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=120, step=1)
    education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
    loan = st.selectbox("Personal Loan", ["yes", "no", "unknown"])
    day_of_week = st.selectbox("Day of Week", ["mon", "tue", "wed", "thu", "fri"])
    emp_var_rate = st.number_input("Empirical Variation Rate", step=0.01)
    euribor3m = st.number_input("Euribor 3 Month Rate", step=0.01)

with col2:
    job = st.selectbox("Job", ["admin.", "blue-collar", "entrepreneur", "housemaid", 
                               "management", "retired", "self-employed", "services", 
                               "student", "technician", "unemployed", "unknown"])
    default = st.selectbox("Default Credit", ["yes", "no", "unknown"])
    contact = st.selectbox("Contact Communication Type", ["cellular", "telephone", "unknown"])
    campaign = st.number_input("Campaign", min_value=0, step=1)
    cons_price_idx = st.number_input("Consumer Price Index", step=0.01)
    nr_employed = st.number_input("Number of Employees", step=1)

with col3:
    marital = st.selectbox("Marital Status", ["married", "single", "divorced", "unknown"])
    housing = st.selectbox("Housing Loan", ["yes", "no", "unknown"])
    month = st.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", 
                                   "aug", "sep", "oct", "nov", "dec"])
    duration = st.number_input("Duration (in seconds)", min_value=0, step=1)
    pdays = st.number_input("Pdays (days since last contact)", min_value=0, step=1)
    previous = st.number_input("Previous (contacts before)", min_value=0, step=1)
    poutcome = st.selectbox("Outcome of previous campaign", ["success", "failure", "nonexistent", "unknown"])
    cons_conf_idx = st.number_input("Consumer Confidence Index", step=0.01)

# Raw input as dict
input_features_raw = {
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "month": month,
    "day_of_week": day_of_week,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome,
    "emp.var.rate": emp_var_rate,
    "cons.price.idx": cons_price_idx,
    "cons.conf.idx": cons_conf_idx,
    "euribor3m": euribor3m,
    "nr.employed": nr_employed
}

input_df = pd.DataFrame([input_features_raw])

# Encode categorical features
categorical_columns = ['job', 'marital', 'education', 'default', 'housing',
                       'loan', 'contact', 'month', 'day_of_week', 'poutcome']

input_df_encoded = pd.get_dummies(input_df, columns=categorical_columns)

# Feature alignment (very important!)
try:
    model_features = model.feature_names_in_
except AttributeError:
    model_features = input_df_encoded.columns

for col in model_features:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0

input_df_encoded = input_df_encoded[model_features]

# Predict
if st.button("Predict", key="predict"):
    with st.spinner("Making prediction..."):
        try:
            prediction = model.predict(input_df_encoded)
            st.success(f"Prediction: {prediction[0]}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

