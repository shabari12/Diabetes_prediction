import numpy as np
import pickle

loaded_model = pickle.load(open('C:/Users/uma digital/Desktop/pythongrind/Diabetes/diabetesmodel.sav','rb'))
scaler = pickle.load(open('C:/Users/uma digital/Desktop/pythongrind/Diabetes/scaler.sav', 'rb'))
m
input_data = (7,196,90,0,0,39.8,0.451,41)

input_data_np = np.asarray(input_data)

input_data_reshaped = input_data_np.reshape(1,-1)
std_data = scaler.transform(input_data_reshaped)
prediction = loaded_model.predict(std_data)
if prediction[0]==0:
  print("The Person is Non Diabetic")
else:
  print("The Person is Diabetic")