import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Pima Diabetes dataset
diabetes_data = pd.read_csv("diabetes.csv")

# Split the dataset into features (X) and target (y)
X = diabetes_data.drop("Outcome", axis=1)
y = diabetes_data["Outcome"]

# Split the dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Define machine learning models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC()
}

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

# Function to predict diabetes
def predict_diabetes(model):
    model.fit(X, y)
    y_pred = model.predict(scaler.transform(pd.DataFrame(user_input)))
    return y_pred[0]

# Prediction and result
st.header("Predictions:")
results = {}
for model_name, model in models.items():
    prediction = predict_diabetes(model)
    results[model_name] = prediction

st.write(results)

# Evaluation metrics and visualization
st.header("Evaluation Metrics:")

for model_name, model in models.items():
    st.subheader(model_name)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)

    st.write(f"Accuracy: {accuracy:.2f}")
    st.write("Confusion Matrix:")
    st.write(cm)

    # Plot the confusion matrix
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", square=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt)

    st.subheader("Classification Report")
    st.text(report)

# Model Comparison
st.header("Model Comparison:")
model_names = list(results.keys())
model_predictions = list(results.values())
st.bar_chart(model_predictions, labels=model_names)

# Footer
st.sidebar.text("Built with Streamlit")
