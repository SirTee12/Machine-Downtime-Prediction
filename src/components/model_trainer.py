import numpy as np
import pandas as pd

import sys
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score
    f1_score
)

from sklearn.ensemble import GradientBoostingClassifier
from src.utils import save_object, evaluate_model
from src.exception import CustomException
from src.logging import logging
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('Artifacts', 'model_trained.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array, target_column_index):
        logging.info('split training and test data set')
        
        try:
            # extract the target column and feature columns dynamically
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1])
                       
            model = GradientBoostingClassifier()
            best_params = {'n_estimators' :  300, 'learning_rate':0.18476368934488233,
                        'max_depth':3, 'subsample': 0.8349830456457842}
            
            model_report:dict = evaluate_model(X_train = X_train, y_train = y_train,
                                               X_test= X_test, y_test= y_test, model = model,
                                               best_params=best_params)
            save_object(
                file_path=self.model_trainer.trained_model_path,
                obj= model
            )
            
            print('\nModel Evaluation Metrics:')
            for metric, value in model_report.items():
                print(f'{metric}: {value:.4f}' if value is not None else f'{metric}: Not Available')
                
            return model_report
        except Exception as e:
            raise CustomException(e, sys)
            
            
            
            
            