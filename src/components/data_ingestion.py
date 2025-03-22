import os
import sys
from src.exception import CustomException
from src.logging import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class dataIngestionConfig:
    train_data_path: str = os.path.join('Artifacts', 'train.csv')
    test_data_path: str = os.path.join('Artifacts', 'test.csv')
    raw_data_path: str = os.path.join('Artifacts', 'raw.csv')
    
class dataIngestion:
    def __init__(self):
        self.ingestion_config = dataIngestionConfig()
        
    def initialize_data_ingestion(self):
        logging.info(f'entered the data ingestion method or component')
        
        try:
            data = pd.read_csv('C:/Users/Administrator/Documents/Data Science Projects/Machine-Downtime-Prediction/data/machine_downtime_cleaned.csv')
            logging.info(f'read dataset as a dataframe')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            data.to_csv(self.ingestion_config.raw_data_path, header = True,
                        index = False)
            stratify_y = data['Downtime']
            logging.info(f'train test split initiated')
            
            train_set, test_set = train_test_split(data, test_size=0.25,
                                                   random_state=42, shuffle=True,
                                                   stratify=stratify_y)
            
            train_set.to_csv(self.ingestion_config.train_data_path, header = False,
                             index = False)
            test_set.to_csv(self.ingestion_config.test_data_path, header = False,
                             index = False)
            
            logging.info(f'data ingestion is completed')
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e, sys)
        
        
if __name__ == '__main__':
    obj = dataIngestion()
    train_data, test_data = obj.initialize_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.initiate_data_tansformation(train_data, test_data)
    
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
