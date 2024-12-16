import numpy as np
import pickle
import streamlit as st
loaded_model = pickle.load(open('C:/Users/uma digital/Desktop/pythongrind/Diabetes/diabetesmodel.sav','rb'))
scaler = pickle.load(open('C:/Users/uma digital/Desktop/pythongrind/Diabetes/scaler.sav', 'rb'))


def prediction(inputdata):
    
   
    input_data_np = np.asarray(inputdata)
    
    input_data_reshaped = input_data_np.reshape(1,-1)
    std_data = scaler.transform(input_data_reshaped)
    prediction = loaded_model.predict(std_data)
    if prediction[0]==0:
        return "The Person is Non Diabetic"
    else:
        return"The Person is Diabetic"


def main():
    st.title("Diabetes Prediction Web App")
    

    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Value")
    SkinThickness = st.text_input("SkinThickness value")
    Insulin = st.text_input("Insulin level")
    BMI = st.text_input("Body Mass Index Value")
    DiabetesPedigreeFunction= st.text_input("Diabetes Pedigree Function value")
    Age = st.text_input("Age")

    diagonis = ''

    if st.button("Diabetes Test Result"):
        diagonis = prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    st.success(diagonis)


if __name__ == '__main__':
    main()