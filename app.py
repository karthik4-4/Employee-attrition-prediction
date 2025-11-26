import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Employee Attrition Prediction", page_icon="💼", layout="centered")
st.title("💼 Employee Attrition Prediction App")
st.write("Enter employee details below to predict whether the employee is likely to leave or stay.")

# Load the model and transformer
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    with open('transformer.pkl', 'rb') as file:
        transformer = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model files: {str(e)}")
    st.stop()

st.subheader("Enter Employee Information")

col1, col2 = st.columns(2)

with col1:
    Age = st.slider("Age", 18, 60, 30)
    JobInvolvement = st.selectbox("Job Involvement (1–4)", [1, 2, 3, 4])
    JobLevel = st.selectbox("Job Level (1–5)", [1, 2, 3, 4, 5])
    MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=50000, step=100, value=5000)
    StockOptionLevel = st.selectbox("Stock Option Level", [0, 1, 2, 3])

with col2:
    TotalWorkingYears = st.slider("Total Working Years", 0, 40, 10)
    YearsAtCompany = st.slider("Years at Company", 0, 40, 5)
    YearsInCurrentRole = st.slider("Years in Current Role", 0, 20, 3)
    YearsWithCurrManager = st.slider("Years with Current Manager", 0, 20, 3)

st.subheader("Select Job Role, Marital Status, and Overtime")

JobRole = st.selectbox(
    "Job Role",
    [
        'Healthcare Representative', 'Human Resources', 'Laboratory Technician', 'Manager',
        'Manufacturing Director', 'Research Director', 'Research Scientist',
        'Sales Executive', 'Sales Representative'
    ]
)

MaritalStatus = st.selectbox("Marital Status", ['Divorced', 'Married', 'Single'])
OverTime = st.selectbox("OverTime", ['No', 'Yes'])

# Create input dataframe
input_data = pd.DataFrame({
    'Age': [Age],
    'JobInvolvement': [JobInvolvement],
    'JobLevel': [JobLevel],
    'JobRole': [JobRole],
    'MaritalStatus': [MaritalStatus],
    'MonthlyIncome': [MonthlyIncome],
    'OverTime': [OverTime],
    'StockOptionLevel': [StockOptionLevel],
    'TotalWorkingYears': [TotalWorkingYears],
    'YearsAtCompany': [YearsAtCompany],
    'YearsInCurrentRole': [YearsInCurrentRole],
    'YearsWithCurrManager': [YearsWithCurrManager]
})

# Prediction button
if st.button("Predict Attrition", type="primary"):
    try:
        transformed_input = transformer.transform(input_data)
        
        # Make prediction
        prediction = model.predict(transformed_input)
        prediction_proba = model.predict_proba(transformed_input)
        
        # Display results
        st.subheader("Prediction Result")
        
        if prediction[0] == 1:
            st.error("⚠️ **High Risk**: This employee is likely to leave the company.")
            st.write(f"**Probability of Attrition**: {prediction_proba[0][1]:.2%}")
        else:
            st.success("✅ **Low Risk**: This employee is likely to stay with the company.")
            st.write(f"**Probability of Retention**: {prediction_proba[0][0]:.2%}")
        
        # Show probability breakdown
        st.write("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Stay Probability", f"{prediction_proba[0][0]:.2%}")
        with col_b:
            st.metric("Leave Probability", f"{prediction_proba[0][1]:.2%}")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.write("Please check that your model and transformer files are compatible with the input data.")
        st.write(f"Input data shape: {input_data.shape}")
        st.write(f"Input columns: {input_data.columns.tolist()}")

# Add footer
st.write("---")
st.caption("💡 This prediction is based on machine learning and should be used as a guide, not a definitive answer.")