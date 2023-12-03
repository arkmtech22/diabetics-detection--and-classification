import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Load Pima dataset
df = pd.read_csv('diabetes.csv')

# Train the classifier
X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Streamlit app
st.title("Diabetes Detection Web App")

# Sidebar with sliders
pregnancies = st.sidebar.slider('Number of Pregnancies', 0, 20, 1)
glucose = st.sidebar.slider('Glucose Level', 0, 200, 100)
blood_pressure = st.sidebar.slider('Blood Pressure', 0, 150, 70)
skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 20)
insulin = st.sidebar.slider('Insulin Level', 0, 846, 79)
bmi = st.sidebar.slider('BMI', 0.0, 67.1, 25.0)
diabetes_pedigree = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
age = st.sidebar.slider('Age', 21, 81, 30)

# Make prediction
if st.button('Predict'):
    user_input = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })
    prediction = clf.predict(user_input)
    st.write(f"Prediction: {'Diabetes' if prediction[0] == 1 else 'No Diabetes'}")

# Evaluation button
if st.button('Evaluate Model'):
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    st.write("Confusion Matrix:")
    st.write(cm)
    st.write("Classification Report:")
    st.write(report)
    st.write(f"Accuracy: {accuracy:.4f}")
    st.write(f"Precision: {precision:.4f}")
    st.write(f"Recall: {recall:.4f}")
    st.write(f"F1 Score: {f1:.4f}")
    st.write(f"ROC-AUC: {roc_auc:.4f}")

