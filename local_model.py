import numpy as np
import pickle

loaded_model = pickle.load(open('C:/Users/uma digital/Desktop/pythongrind/Diabetes/diabetesmodel.sav','rb'))
scaler = pickle.load(open('C:/Users/uma digital/Desktop/pythongrind/Diabetes/scaler.sav', 'rb'))
# Making a predictive System
input_data = (7,196,90,0,0,39.8,0.451,41)
# changing input data to numpy array for easy understanding of the computer processing is efficient
input_data_np = np.asarray(input_data)
# reshaping the np array as predicting for only one instance
input_data_reshaped = input_data_np.reshape(1,-1)
std_data = scaler.transform(input_data_reshaped)
prediction = loaded_model.predict(std_data)
if prediction[0]==0:
  print("The Person is Non Diabetic")
else:
  print("The Person is Diabetic")