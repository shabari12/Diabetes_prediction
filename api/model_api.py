from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

app = FastAPI()


class ModelInput(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# loading the saved model


model = pickle.load(open('C:/Users/uma digital/Desktop/pythongrind/Diabetes/diabetesmodel.sav','rb'))
#scaler = pickle.load(open('C:/Users/uma digital/Desktop/pythongrind/Diabetes/scaler.sav', 'rb'))    

@app.post('/diabetes_predictions')
def prediction(input_params:ModelInput):
    input_data = input_params.json()
    input_dict = json.loads(input_data)
    preg = input_dict['Pregnancies']
    glu = input_dict['Glucose']
    bp = input_dict['BloodPressure']
    st = input_dict['SkinThickness']
    ins = input_dict['Insulin']
    bmi = input_dict['BMI']
    dpf = input_dict['DiabetesPedigreeFunction']
    age = input_dict['Age']
    

    input_list = [preg,glu,bp,st,ins,bmi,dpf,age]

    prediction = model.predict([input_list])

    if prediction[0] == 0:
        return "The prerson is not Diabetic"
    else:
        return "The person is Diabetic"