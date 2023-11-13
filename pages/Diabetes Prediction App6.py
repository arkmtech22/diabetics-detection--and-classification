            
# Import necessary modules
import streamlit as st


# Create a Streamlit app
st.title("Pima Diabetes Prediction App")

# Sidebar with sliders for user input
st.sidebar.header("Enter Patient Information")

pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
glucose = st.sidebar.slider("Glucose", 0, 199, 117)
blood_pressure = st.sidebar.slider("Blood Pressure", 0, 122, 72)
skin_thickness = st.sidebar.slider("Skin Thickness", 0, 99, 23)
insulin = st.sidebar.slider("Insulin", 0, 846, 30)
bmi = st.sidebar.slider("BMI", 0.0, 67.1, 32.0)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.42, 0.3725)
age = st.sidebar.slider("Age", 21, 81, 29)

# Create a dictionary for user input
user_input = {
    "Pregnancies": [pregnancies],
    "Glucose": [glucose],
    "BloodPressure": [blood_pressure],
    "SkinThickness": [skin_thickness],
    "Insulin": [insulin],
    "BMI": [bmi],
    "DiabetesPedigreeFunction": [dpf],
    "Age": [age]
}

# Display user input data
st.sidebar.subheader("User Input Data:")
st.sidebar.write(user_input)


import streamlit as st
import pandas as pd

# Load the Pima Indian Diabetes dataset
df = pd.read_csv('diabetes.csv')

# Define the number of rows to display at a time
rows_to_display = 10

# Define a variable to keep track of the current page
page = st.sidebar.number_input('Page', min_value=1, max_value=(len(df) - 1) // rows_to_display + 1, value=1)

# Calculate the start and end indices for the current page
start_idx = (page - 1) * rows_to_display
end_idx = min(page * rows_to_display, len(df))

# Display the dataset for the current page
st.write(f"Displaying rows {start_idx + 1} to {end_idx} of {len(df)}")
st.write(df.iloc[start_idx:end_idx])

# Allow users to navigate between pages
if st.button("Previous Page", key="previous"):
    page = max(1, page - 1)
if st.button("Next Page", key="next"):
    page = min((len(df) - 1) // rows_to_display + 1, page + 1)




