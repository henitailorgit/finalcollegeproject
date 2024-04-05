import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


st.title("Experience Page")

# Assuming you have a DataFrame with the following features:
# "Acceleration Value", "Abnormal Short Term Variability Value", "Mean Value of Short Term Variability",
# "Time % with Abnormal Long Term Variability Value", "Histogram Mean Value", "Temperature", "Conditions", "Histom", "HOLONCLLO", "B", "IN"
features = ["Acceleration Value", "Abnormal Short Term Variability Value", "Mean Value of Short Term Variability", "Time % with Abnormal Long Term Variability Value", "Histogram Mean Value", "Temperature"]
target = "Fetal Health"

# Load the dataset
data = pd.read_csv(r"C:\Users\HENI TAILOR\Fetal Health Classification Project\fetal_health.csv")


# Prepare the data
X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a random forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict fetal health based on given values
given_values = pd.DataFrame([{"Acceleration Value": 0.06,
                            "Abnormal Short Term Variability Value": 17,
                            "Mean Value of Short Term Variability": 26,
                            "Time % with Abnormal Long Term Variability Value": 43,
                            "Histogram Mean Value": 137,
                            "Temperature": 23.0}])

prediction = model.predict(given_values)

print(f"Predicted Fetal Health: {prediction[0]}")

# To check the accuracy of the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")







##################################################################################
                               #Second Prediction#
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


# Load the dataset
data = pd.read_csv("C:/Users/HENI TAILOR/Fetal Health Classification Project/fetal_health.csv")

# Separate the features and the target variable
X = data.drop("heni", axis=1)
y = data["y_test"]

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the training and testing sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model using random forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the model and scaler
joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")

# Load the saved model and scaler
loaded_model = joblib.load("model.joblib")
loaded_scaler = joblib.load("scaler.joblib")

# Load the new dataset
new_data = pd.read_csv("new_data.csv")

# Standardize the new data
new_data_scaled = loaded_scaler.transform(new_data)

# Make predictions on the new data using the saved model
predictions = loaded_model.predict(new_data_scaled)

# Print the predictions
print(predictions)
