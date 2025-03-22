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
from src.utils import save_object

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
                            'Spindle_Vibration', 'Tool_Vibration', 'Torque',
                             'Hydraulic_Pressure', 'Coolant_Pressure',
                             'Cutting', 'Spindle_Speed']
            
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
            
    
    def initiate_data_tansformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info(f'Read train and test data completed')
            logging.info(f'obtaining the preprocessor object')
            
            preprocessor_obj = self.get_data_transformer_object()
            
            target_col_name = 'Downtime'
            Label_Encoder = LabelEncoder()
            
            # get the train features and target variables and apply label encoding to the 
            # teh target variable
            input_feature_train = train_df.drop(columns= [
                target_col_name, 'Air_System_Pressure', 
                'Spindle_Bearing_Temperature', 'Voltage'
            ], axis = 1)
            
            target_feature_train = Label_Encoder.fit_transform(train_df[target_col_name])

            # get the test features and target variables and apply label encoding to the 
            # teh target variable 
            input_feature_test = test_df.drop(columns= [
                target_col_name, 'Air_System_Pressure', 
                'Spindle_Bearing_Temperature', 'Voltage'
            ], axis = 1)       
            
            target_feature_test = Label_Encoder.transform(test_df[target_col_name])
            
            logging.info(f'Applying the preprocesing objec to training and test data')
            
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train)
            input_feature_test_arr = preprocessor_obj.fit_transform(input_feature_test)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test)
            ]
            
            logging.info(f'Data transformation complete')
            logging.info(f'saved preprocessing object')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
            
        except Exception as e:
            raise CustomException(e, sys)
            
            
