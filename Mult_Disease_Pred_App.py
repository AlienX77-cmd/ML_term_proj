import pickle
import streamlit as st
from streamlit_option_menu import option_menu

import numpy as np

# loading the saved model
multiple_disease_pred_model = pickle.load(open('RF_model.sav', 'rb'))

# sidebar for navigate
with st.sidebar:
    selected = option_menu('Multiple Diseases Prediction System', ['3 Diseases Prediction'], default_index=0)

# Diabetes, Cardiovascular and Chronic Kidney Disease Prediction System
if (selected == '3 Diseases Prediction'):

    # page title
    st.title('Diabetes, Cardiovascular and Chronic Kidney Disease Prediction System')

    # getting the input data from user
    col1, col2, col3, col4 = st.columns(4)

    # 1) age
    with col1: Age = st.text_input("Age", placeholder="years")

    # 2) bp
    with col2: Blood_Pressure = st.text_input("Blood Pressure", placeholder="mm/Hg")

    # 3) sg
    with col3: Specific_Gravity = st.text_input("Specific Gravity", placeholder="sg (relative density)")

    # 4) al
    with col4: Albumin = st.text_input("Albumin", placeholder="g/dL")

    # 5) bgr
    with col1: Blood_Glucose_Random = st.text_input("Blood Glucose", placeholder="mgs/dl")

    # 6) bu
    with col2: Blood_Urea = st.text_input("Blood Urea", placeholder="mgs/dl")

    # 7) sc
    with col3: Serum_Creatinine = st.text_input("Serum Creatinine", placeholder="mgs/dl")

    # 8) sod
    with col4: Sodium = st.text_input("Sodium", placeholder="mEq/L")

    # 9) hemo
    with col1: Hemoglobin = st.text_input("Hemoglobin", placeholder="hemo in gms")

    # 10) pcv
    with col2: Packed_Cell_Voume = st.text_input("Packed Cell Volume", placeholder="in percentage")

    # 11) rc
    with col3: Red_Blood_Cell_Count = st.text_input("Red Blood Cell", placeholder="millions/cmm")

    # 12) htn
    with col4: Hypertension = st.text_input("Hypertension", placeholder="yes = 1 | no = 0")

    # 13) appet
    with col1: Appetite = st.text_input("Appetite", placeholder= "good = 1 | poor = 0")

    # 14) pe
    with col2: Pedal_Edema = st.text_input("Pedal Edema", placeholder="yes = 1 | no = 0")

    # Data preparation
    a = [Age, Blood_Pressure, Specific_Gravity, 
        Albumin, Blood_Glucose_Random, Blood_Urea, 
        Serum_Creatinine, Sodium, Hemoglobin,
        Packed_Cell_Voume, Red_Blood_Cell_Count,
        Hypertension, Appetite, Pedal_Edema]

    # Convert inputs to float, ensuring all strings are valid numbers
    try:
        b = [float(i) if i != '' else None for i in a]
    except ValueError as e:
        st.error(f"Invalid input: {e}")

    # Ensure that all inputs are provided before making a prediction
    if None not in b:
        input_numpy = np.asarray(b)

        input_reshaped = input_numpy.reshape(1,-1)

        # Code for Prediction (Classification) System
        Disease_Diagnosis = ''

        # Creating a button for Prediction (Classification)

        if st.button("Disease Test Results"):
            Disease_Prediction = multiple_disease_pred_model.predict(input_reshaped)
            if (Disease_Prediction[0] == 0): Disease_Diagnosis = "This patient has no disease."
            elif (Disease_Prediction[0] == 1): Disease_Diagnosis = "This patient has Chronic Kidney disease."
            elif (Disease_Prediction[0] == 2): Disease_Diagnosis = "This patient has Cardiovascular disease."
            elif (Disease_Prediction[0] == 3): Disease_Diagnosis = "This patient has Cardiovascular and Chronic Kidney disease."
            elif (Disease_Prediction[0] == 4): Disease_Diagnosis = "This patient has Diabetes disease."
            elif (Disease_Prediction[0] == 5): Disease_Diagnosis = "This patient has Diabetes and Chronic Kidney disease."
            elif (Disease_Prediction[0] == 6): Disease_Diagnosis = "This patient has Diabetes and Cardiovascular disease."
            elif (Disease_Prediction[0] == 7): Disease_Diagnosis = "This patient has all diseases (Diabetes, Cardiovascular, and Chronic Kidney disease)."
            else: Disease_Diagnosis = "An Error has occured in the prediction process."

        st.success(Disease_Diagnosis)

    else:
        st.error("Please fill out all the fields before proceeding.")

# Footer
st.write('---')  # Draw a line to separate the footer
st.markdown("""
    &copy; 2024 Kittipak Wibulsthien (6310505688) - Machine Learning Term Project. All Rights Reserved.
    """, unsafe_allow_html=True)