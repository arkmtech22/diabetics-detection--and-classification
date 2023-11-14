            
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

# Upload the Pima Indian Diabetes dataset
st.sidebar.header("Upload Pima Indian Diabetes Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Check if a file was uploaded
if uploaded_file is not None:
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the dataset
    st.write("Pima Indian Diabetes Dataset")
    st.write(df)
else:
    st.sidebar.warning("Upload a CSV file to load the dataset.")


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

#import streamlit as st
#import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
#from sklearn.feature_selection import SelectFromModel

# Load the Pima Indian Diabetes dataset
st.sidebar.header("Upload Pima Indian Diabetes Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Check if a file was uploaded
if uploaded_file is not None:
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(uploaded_file)

    # Display dataset description
    st.write("## Dataset Description")
    st.write(df.describe())

    # Display feature importance
    st.write("## Feature Importance")
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Train a Random Forest model for feature importance
   # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   # rf_classifier = RandomForestClassifier()
    #rf_classifier.fit(X_train, y_train)

    # Display feature importance plot
  #  feature_importance = pd.Series(rf_classifier.feature_importances_, index=X.columns)
  #  feature_importance = feature_importance.sort_values(ascending=False)
  #  st.bar_chart(feature_importance)

    # Display attribute-wise graphs
    st.write("## Attribute-wise Graphs")
    for column in df.columns:
        if column != "Outcome":
            st.write(f"### {column} Distribution")
            plt.figure(figsize=(8, 6))
            if df[column].dtype == 'float64' or df[column].dtype == 'int64':
                sns.histplot(df[column], kde=True)
            else:
                sns.countplot(x=column, data=df, hue="Outcome")
            st.pyplot()

    # Display heatmap
    st.write("## Heatmap")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=.5)
    st.pyplot()

    # Display other required graphs and visualizations as needed

else:
    st.sidebar.warning("Upload a CSV file to load the dataset.")


