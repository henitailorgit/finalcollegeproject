import streamlit as st
import pickle
import joblib
from PIL import Image
from pathlib import Path
#import os
import numpy as np

# Load the model
#model = joblib.load('D:/Users/HENI TAILOR/Fetal Health Classification Project/Model/ML_Model1.pkl')
#Get the correct file path using expanduser and abspath
#model_path = os.path.expanduser(r'D:/Users/HENI TAILOR/Fetal Health Classification Project/Model/ML_Model1.pkl','rb')
#model_file = os.path.abspath(model_path)

# Load the model
with open('D:/Users/HENI TAILOR/Fetal Health Classification Project/Model/ML_Model1.pkl', 'rb') as f:
    model = pickle.load(f)

def run():
    st.title("Fetal Health Classification")

    img1 = Image.open('D:/Users/HENI TAILOR/Fetal Health Classification Project/Frontend(Streamlit Project)/Pages/Baby.jpg')
    img1 = img1.resize((156,145))
    st.image(img1, use_column_width=False)
    
    acceleration_value = st.number_input("Acceleration Value", value = 0)
    abnormal_short_term_variability_value = st.number_input('Abnormal Short Term Variability Value', value = 0)
    mean_value_of_short_term_variability = st.number_input('Mean Value of Short Term Variability', value = 0)
    time_with_abnormal_long_term_variability_value = st.number_input("Time % with Abnormal Long Term Variability Value",value = 0)
    histogram_mean_value = st.number_input("Histogram Mean Value", value = 0)
    temperature = st.number_input("Temperature", value = 0)

    dur_display = ['2 Month','4 Month','6 Month','9 Month']
    dur_options = range(len(dur_display))
    dur = st.selectbox("Pregnancy Month", dur_options, format_func = lambda x:dur_display[x])


    if st.button("Predict"):
        duration = 0
        if dur == '6 Month':
            duration = 60
        if dur == '4 Month':
            duration = 180
        if dur == '2 Month':
            duration = 240
        if dur == '9 Month':
            duration = 360

        features = [[acceleration_value, abnormal_short_term_variability_value, mean_value_of_short_term_variability,
                     time_with_abnormal_long_term_variability_value, histogram_mean_value,
                     temperature, duration]]

        st.write("Features:", features)

        # Predict fetal health
        prediction = model.predict(features)

        # Display the prediction result
        if np.round(model.predict_proba(features)[0][1], 2) >= 0.96:  # Check model accuracy
            st.success(
                "Hello: [Your Name] ||"
                "Pregnancy: {} || "
                "According to Checkup, Your Pregnancy report is Abnormal".format(acceleration_value)
            )
        else:
            st.error(
                "Hello: [Your Name] ||"
                "Pregnancy: {} || "
                "According to Checkup, Your Pregnancy report is Normal".format(acceleration_value)
            )

run()
