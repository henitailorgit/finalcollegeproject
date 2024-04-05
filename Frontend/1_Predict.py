import numpy as np
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the trained Random Forest classifier (assuming 'predict_model.joblib' is in the same directory)
rf_classifier = joblib.load('../predict_model.joblib')

# Define the class labels
class_labels = ['Normal', 'Suspect', 'Pathological']

# Define the function to predict fetal health
def predict_fetal_health(input_data):
    """Predicts fetal health based on input features.

    Args:
        input_data (dict): Dictionary containing key-value pairs for features.

    Returns:
        str: Predicted fetal health classification.
    """

    # Convert input dictionary to array
    input_array = np.array(list(input_data.values())).reshape(1, -1)

    # Make sure the input data has the correct number of features
    if input_array.shape[1] != 17:
        st.error("Error: Input data must contain all 17 features used during training.")
        return None

    # Scale the input features
    scaler = StandardScaler()
    input_array_scaled = scaler.fit_transform(input_array)

    # Make prediction
    prediction = int(rf_classifier.predict(input_array_scaled)[0])

    # Map prediction to class label
    fetal_health = class_labels[prediction]

    # Return predicted fetal health
    return fetal_health

# Define the Streamlit app
def main():
    """Runs the Streamlit app for fetal health prediction."""

    # Title and subheader
    st.title("Fetal Health Prediction")
    st.subheader("Enter the values for the following features:")

    # Create input fields for features
    input_data = {}
    num_features = 17  # Assuming you have 17 features
    for i in range(1, num_features + 1):
        feature_name = f'Feature {i}'
        input_data[feature_name] = st.number_input(feature_name, key=feature_name)

    # Make prediction button
    if st.button("Predict"):
        try:
            # Call prediction function
            fetal_health_prediction = predict_fetal_health(input_data)
            if fetal_health_prediction is not None:
                st.write("The predicted fetal health classification is:", fetal_health_prediction)
        except Exception as e:
            # Handle errors gracefully
            st.error("An error occurred during prediction. Please check your input values and try again.")
            print(f"Error: {e}")  # Log error for debugging (optional)

# Run the Streamlit app
if __name__ == '__main__':
    main()