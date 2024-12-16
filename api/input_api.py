import json
import requests

url = 'http://127.0.0.1:8000/diabetes_predictions'

input_data = {
    'Pregnancies': 2,  
    'Glucose': 130,  
    'BloodPressure': 80,  
    'SkinThickness': 20,  
    'Insulin': 85,  
    'BMI': 28.5,  
    'DiabetesPedigreeFunction': 0.5,  
    'Age': 45  
}


response = requests.post(url, json=input_data)

print(response.text)
