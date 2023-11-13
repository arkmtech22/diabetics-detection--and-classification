import streamlit as st
import pandas as pd
import numpy as np
pip install scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
pip install scikit-fuzzy
import numpy as np
import skfuzzy as fuzz
 # import skfuzzy as fuzz
 # Load the diabetes dataset
 # diabetes_df = pd.read_csv("/content/drive/My Drive/COLAB/diabetes.csv")
diabetes_df = pd.read_csv(r'diabetes.csv')

 # Split the dataset into features and labels
X = diabetes_df.iloc[:, :-1].values
y = diabetes_df.iloc[:, -1].values

 # Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

 # Feature scaling using the StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

 # Binary classification using logistic regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
acc_logreg = accuracy_score(y_test, y_pred_logreg)
cr_logreg = classification_report(y_test, y_pred_logreg)
cm_logreg = confusion_matrix(y_test, y_pred_logreg)

 # Multiclass classification using decision tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)
cr_dt = classification_report(y_test, y_pred_dt)
cm_dt = confusion_matrix(y_test, y_pred_dt)

 # Rule-based classification using a fuzzy expert system
def get_fuzzy_expert_system(X_train, y_train, X_test):
    # Divide the output space into three fuzzy sets
    diabetes = fuzz.trimf([0, 0, 1], [0, 0.5, 1])
    prediabetes = fuzz.trimf([0, 1, 2], [0.5, 1, 1.5])
    normal = fuzz.trimf([1, 1, 2], [1, 1.5, 2])

     # Define the fuzzy rules
    rule1 = np.fmax(diabetes, normal)
    rule2 = prediabetes
    rule3 = np.fmax(prediabetes, diabetes)

     # Combine the rules using the OR operator
    or_rule = np.fmax(rule1, np.fmax(rule2, rule3))

     # Compute the output using the weighted average defuzzification method
    y_train_fuzzy = np.column_stack([fuzz.interp_membership(X_train[:, i], X_train[:, i], or_rule[i]) for i in range(len(X_train[0]))])
    y_test_fuzzy = np.column_stack([fuzz.interp_membership(X_test[:, i], X_test[:, i], or_rule[i]) for i in range(len(X_test[0]))])
    y_pred_fuzzy = np.argmax(y_train_fuzzy, axis=1)
    y_pred_fuzzy_test = np.argmax(y_test_fuzzy, axis=1)

    return y_pred_fuzzy_test
y_pred_fuzzy = get_fuzzy_expert_system(X_train, y_train, X_test)
acc_fuzzy = accuracy_score(y_test, y_pred_fuzzy)
cr_fuzzy = classification_report(y_test, y_pred_fuzzy)
cm_fuzzy = confusion_matrix(y_test, y_pred_fuzzy)

 # Fuzzy logic-based classification using fuzzy c-means clustering
def get_fuzzy_cmeans(X_train, y_train, X_test):
     # Use the elbow method to determine the number of clusters
    wcss = []
    for i in range(1, 11):	
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X_train)
        wcss.append(kmeans.inertia_)
    st.line_chart(wcss)
    k = st.slider('Select number of clusters for fuzzy c-means:', min_value=1, max_value=10)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X_train.T, k, 2, error=0.005, maxiter=1000, init=None)
    u_pred = fuzz.cluster.cmeans_predict(X_test.T, cntr, 2, error=0.005, maxiter=1000)
    y_pred_fcm = np.argmax(u_pred, axis=0)
    return y_pred_fcm

y_pred_fcm = get_fuzzy_cmeans(X_train, y_train, X_test)
y_pred_fuzzy_test = np.argmax(y_pred_fcm, axis=0)
acc_fcm = accuracy_score(y_test, y_pred_fuzzy_test)
cr_fcm = classification_report(y_test, y_pred_fuzzy_test)
cm_fcm = confusion_matrix(y_test, y_pred_fuzzy_test)

 # Display the results
st.write('## Diabetes Prediction with Supervised Machine Learning')
st.write('### Binary Classification using Logistic Regression')
st.write('Accuracy:', acc_logreg)
st.write('Classification Report:\n', cr_logreg)
st.write('Confusion Matrix:\n', cm_logreg)
st.write('### Multiclass Classification using Decision Tree')
st.write('Accuracy:', acc_dt)
st.write('Classification Report:\n', cr_dt)
st.write('Confusion Matrix:\n', cm_dt)
st.write('### Rule-based Classification using a Fuzzy Expert System')
st.write('Accuracy:', acc_fuzzy)
st.write('Classification Report:\n', cr_fuzzy)
st.write('Confusion Matrix:\n', cm_fuzzy)
st.write('### Fuzzy Logic-based Classification using Fuzzy C-means Clustering')
st.write('Accuracy:', acc_fcm)
st.write('Classification Report:\n', cr_fcm)
st.write('Confusion Matrix:\n', cm_fcm)
            
