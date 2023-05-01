import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = 'artifacts/proprocessor.pkl'
            model_path = 'artifacts/model.pkl'
            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)
            data_scaled = preprocessor.transform(features)
            pred = model.predict(data_scaled)
            return pred
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        Age: int,
        Diabetes: str,
        BloodPressureProblems: str,
        AnyTransplants: str,
        AnyChronicDiseases: str,
        Height: float,
        Weight: float,
        KnownAllergies: str,
        HistoryOfCancerInFamily: str,
        NumberOfMajorSurgeries: str,
        ):

        self.Age = Age

        self.Diabetes = Diabetes

        self.BloodPressureProblems = BloodPressureProblems

        self.AnyTransplants = AnyTransplants

        self.AnyChronicDiseases = AnyChronicDiseases

        self.Height = Height

        self.Weight = Weight
        self.KnownAllergies = KnownAllergies
        self.HistoryOfCancerInFamily = HistoryOfCancerInFamily
        self.NumberOfMajorSurgeries = NumberOfMajorSurgeries

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "Diabetes": [self.Diabetes],
                "BloodPressureProblems": [self.BloodPressureProblems],
                "AnyTransplants": [self.AnyTransplants],
                "AnyChronicDiseases": [self.AnyChronicDiseases],
                "Height": [self.Height],
                "Weight": [self.Weight],
                "KnownAllergies": [self.KnownAllergies],
                "HistoryOfCancerInFamily": [self.HistoryOfCancerInFamily],
                "NumberOfMajorSurgeries": [self.NumberOfMajorSurgeries]
            }
            df = pd.DataFrame(custom_data_input_dict)
            df = df[['Age', 'Diabetes', 'BloodPressureProblems',
       'AnyTransplants', 'AnyChronicDiseases', 'Height', 'Weight',
       'KnownAllergies', 'HistoryOfCancerInFamily', 'NumberOfMajorSurgeries']]
            logging.info('Dataframe Gathered')
            return df

            

        except Exception as e:
            raise CustomException(e, sys)
        

