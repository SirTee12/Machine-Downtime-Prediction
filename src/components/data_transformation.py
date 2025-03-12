import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, RobustScaler, \
                                OneHotEncoder, LabelEncoder
                                
from src.exception import CustomException
from src.logging import logging
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer

@dataclass

class DataTransformationConfig():
    '''
    Created a class to store the the file path for the preprocessor implementation file 
    '''
    
    preprocessor_obj_file_path = os.path.join('Artifacts', 'Preprocessor.pkl')
    

class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try: 
            numeric_col = ['Coolant_Temperature','Hydraulic_Oil_Temperature',
                            'Spindle_Vibration', 'Tool_Vibration', 'Torque(Nm)',
                             'Hydraulic_Pressure(Pa)', 'Coolant_Pressure(Pa)',
                             'Cutting(N)', 'Spindle_Speed(RPS)']
            
            num_pipeline  = Pipeline(
                
                steps = [
                    ('imputer', KNNImputer(n_neighbors=3)),
                    ('Robust Scaler', RobustScaler()),
                    ('Standard Scaler', StandardScaler())
                ]
            )
            
            logging.info(f'Numerical Column {numeric_col}')
            
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numeric_col)
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def apply_label_encoding(self, df, target_col):
        
        '''
        Apply label encoding to the target variable
        '''
        try: 
            label_encoder = LabelEncoder()
            df[target_col] = label_encoder.fit_transform(df[target_col])
        
            # print log information
            logging.info(f'Label encoding applied on target variable')
            return df, label_encoder
        except Exception as e:
            raise CustomException(e, sys)
            
    
