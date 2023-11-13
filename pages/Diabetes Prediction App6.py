import tkinter as tk
from tkinter import ttk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Create the main application window
app = tk.Tk()
app.title("Diabetes Prediction App")

# Function to create a slider with specified label, start, and end values
def create_slider(parent, label, start, end):
    label_widget = ttk.Label(parent, text=label)
    label_widget.pack()
    slider = ttk.Scale(parent, from_=start, to=end, orient="horizontal")
    slider.pack()
    value_label = ttk.Label(parent, text=f"{label}: {start}")  # Label to display the current value
    value_label.pack()
    
    # Function to update the value label
    def update_value(value):
        value_label.config(text=f"{label}: {slider.get()}")
    
    # Bind the slider to the update function
    slider.config(command=lambda value=slider: update_value(value))

    return slider

# Create sliders for user input based on the Pima Indian Diabetes dataset
pregnancies_slider = create_slider(app, "Pregnancies", 0, 17)
glucose_slider = create_slider(app, "Glucose", 0, 199)
blood_pressure_slider = create_slider(app, "Blood Pressure", 0, 122)
skinthickness_slider = create_slider(app, "Skin Thickness", 0, 99)  # Corrected slider name
insulin_slider = create_slider(app, "Insulin", 0, 846)
bmi_slider = create_slider(app, "BMI", 0, 67.1)
dpf_slider = create_slider(app, "Diabetes Pedigree Function", 0.078, 2.42)
age_slider = create_slider(app, "Age", 21, 81)

# Create radio buttons for selecting "Diabetes" or "No Diabetes"
diabetes_var = tk.StringVar()
diabetes_var.set("Diabetes")  # Default selection
diabetes_radio_button1 = ttk.Radiobutton(app, text="Diabetes", variable=diabetes_var, value="Diabetes")
diabetes_radio_button2 = ttk.Radiobutton(app, text="No Diabetes", variable=diabetes_var, value="No Diabetes")
diabetes_radio_button1.pack()
diabetes_radio_button2.pack()

# Define empty lists to store user inputs
feature_data = []
labels = []

def collect_data():
    # Get user input values from sliders and radio buttons
    glucose = glucose_slider.get()
    insulin = insulin_slider.get()
    bp = blood_pressure_slider.get()
    bmi = bmi_slider.get()
    dpf = dpf_slider.get()
    age = age_slider.get()
    pregnancies = pregnancies_slider.get()
    skinthickness = skinthickness_slider.get()
    
    # Convert the selected label to a binary value (0 for No Diabetes, 1 for Diabetes)
    label = diabetes_var.get()
    
    # Append the user inputs and label to the respective lists
    feature_data.append([glucose, insulin, bp, bmi, dpf, age, pregnancies, skinthickness])
    labels.append(label)
    
    # Reset sliders and radio button
    glucose_slider.set(0)
    insulin_slider.set(0)
    blood_pressure_slider.set(0)
    bmi_slider.set(0)
    dpf_slider.set(0)
    age_slider.set(0)
    pregnancies_slider.set(0)
    skinthickness_slider.set(0)
    diabetes_var.set("Diabetes" if diabetes_var.get() == "No Diabetes" else "No Diabetes")
### Function to collect user data
##def collect_data():
##    # Get user input values from sliders and radio buttons
##    glucose = glucose_slider.get()
##    insulin = insulin_slider.get()
##    bp = blood_pressure_slider.get()
##    bmi = bmi_slider.get()
##    dpf = dpf_slider.get()
##    age = age_slider.get()
##    pregnancies = pregnancies_slider.get()
##    skinthickness = skinthickness_slider.get()
##    
##    # Convert the selected label to a binary value (0 for No Diabetes, 1 for Diabetes)
##    label = 1 if diabetes_var.get() == "Diabetes" else 0
##
##    # Append the user inputs and label to the respective lists
##    feature_data.append([glucose, insulin, bp, bmi, dpf, age, pregnancies, skinthickness])
##    labels.append(label)
##
##    # Reset sliders and radio button
##    glucose_slider.set(0)
##    insulin_slider.set(0)
##    blood_pressure_slider.set(0)
##    bmi_slider.set(0)
##    dpf_slider.set(0)
##    age_slider.set(0)
##    pregnancies_slider.set(0)
##    skinthickness_slider.set(0)
##    diabetes_var.set("Diabetes")

# Function to predict diabetes
def predict_diabetes():
    if len(feature_data) < 2:
        result_label.config(text="Collect more data points for splitting and prediction.")
        return
    
    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.2, random_state=42)

    # Train a simple Logistic Regression model (you can replace this with other algorithms)
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "K-Nearest Neighbors": KNeighborsClassifier()
    }

    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = accuracy

    # Plot the prediction results
    plt.bar(range(len(results)), list(results.values()), align='center')
    plt.xticks(range(len(results)), list(results.keys()), rotation='vertical')
    plt.ylabel('Accuracy')
    plt.title('Diabetes Prediction Results')
    plt.show()

    # Clear feature_data and labels
    feature_data.clear()
    labels.clear()

# Button to collect data
collect_data_button = ttk.Button(app, text="Collect Data", command=collect_data)
collect_data_button.pack()

# Button to predict diabetes
predict_button = ttk.Button(app, text="Predict Diabetes", command=predict_diabetes)
predict_button.pack()

# Label to display the prediction results
result_label = ttk.Label(app, text="")
result_label.pack()

# Start the main event loop
app.mainloop()
