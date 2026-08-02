import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# --- Page Configuration ---
st.set_page_config(page_title="SalaryIQ", page_icon="💼", layout="centered")

st.title("💼 SalaryIQ")
st.write("Enter candidate details to predict a fair salary.")

# --- Train Model ---
@st.cache_resource
def train_model():
    data = {
        'Experience_Years': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Written_Test_Score': [7, 8, 6, 9, 7, 8, 9, 8, 10, 9],
        'Interview_Score': [6, 7, 7, 8, 6, 8, 9, 9, 10, 10],
        'Salary_INR': [450000, 500000, 550000, 650000, 680000, 720000, 800000, 850000, 1600000, 1000000]
    }
    df = pd.DataFrame(data)
    X = df[['Experience_Years', 'Written_Test_Score', 'Interview_Score']]
    y = df['Salary_INR']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

# --- User Inputs ---
exp = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
written = st.number_input("Written Test Score (0-10)", min_value=0.0, max_value=10.0, value=5.0, step=1.0)
interview = st.number_input("Interview Score (0-10)", min_value=0.0, max_value=10.0, value=5.0, step=1.0)

# --- Prediction Action ---
if st.button("Calculate Salary", type="primary"):
    if written == 0 and interview == 0:
        st.error("ERROR: Minimum test marks are needed.")
    else:
        input_data = pd.DataFrame([[exp, written, interview]], 
                                  columns=['Experience_Years', 'Written_Test_Score', 'Interview_Score'])
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Salary: ₹{prediction:,.2f}")
