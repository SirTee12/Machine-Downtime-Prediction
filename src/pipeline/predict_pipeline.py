import sys
import os
import pandas as pd
import numpy as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = os.path.join('Artifacts', 'model.pkl')
            preprocessor_path = os.path.join('Artifacts', 'preprocessor.pkl')
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self,
                 Coolant_Temperature:float,
                 Hydraulic_Oil_Temperature:float,
                 Spindle_Vibration:float, 
                 Tool_Vibration:float, 
                 Torque:float,
                 Hydraulic_Pressure:float, 
                 Coolant_Pressure:float,
                 Cutting:float, 
                 Spindle_Speed:float
                 ):
        self.Coolant_Temperature = Coolant_Temperature
        self.Hydraulic_Pressure = Hydraulic_Pressure
        self.Hydraulic_Oil_Temperature = Hydraulic_Oil_Temperature
        self.Spindle_Vibration = Spindle_Vibration
        self.Tool_Vibration = Tool_Vibration
        self.Torque = Torque
        self.Coolant_Pressure = Coolant_Pressure
        self.Cutting = Cutting
        self.Spindle_Speed = Spindle_Speed
        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Coolant_Temperature':self.Coolant_Temperature,
                'Hydraulic_Pressure':self.Hydraulic_Pressure,
                'Hydraulic_Oil_Temperature':self.Hydraulic_Oil_Temperature,
                'Spindle_Vibration':self.Spindle_Vibration,
                'Tool_Vibration':self.Tool_Vibration,
                'Torque':self.Torque,
                'Coolant_Pressure':self.Coolant_Pressure,
                'Cutting':self.Cutting,
                'Spindle_Speed':self.Spindle_Speed
            }
            return custom_data_input_dict
        
        except Exception as e:
            raise CustomException(e, sys)
