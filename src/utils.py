import os
import sys

import pandas as pd
import numpy as np 
import pickle
#import dill

from src.logging import logging


from sklearn.model_selection import StratifiedKFold, GridSearchCV
from src.exception import CustomException
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score


def save_object(file_path, obj):
    
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, 
                   model, best_params):
    """
    Evaluates machine learning models using Precision, Recall, F1-score, and ROC-AUC.

    Parameters:
    - X_train, y_train: Training data and labels
    - X_test, y_test: Test data and labels
    - models: Dictionary of models { 'model_name': model_instance }
    - best_params: Dictionary of best parameters { 'model_name': best_params_dict }

    Returns:
    - report: Dictionary with evaluation metrics for each model
    """
    report = {}
    
    try:
        logging.info(f'training and evaluating the model.....')
        
        # set the best hyperparameters
        model.set_params(**best_params)
        
        #Train the model
        model.fit(X_train, y_train)
        
        #predictions
        y_pred = model.predict(X_test)
        
        # compute the evaluation metrics 
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1_scores = f1_score(y_test, y_pred)
        
        # # ROC-AUC requires probability estimates
        if hasattr(model, 'predict_proba'):
            pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc_scores = roc_auc_score(y_test, pred_proba)
            
        else:
            roc_auc_scores = None # some models do not support probability prediction
            
        
        # store result in report
        report = {
            'precison':precision,
            'recall':recall,
            'f1_score':f1_scores,
            'roc_auc':roc_auc_scores
        }
        
        logging.info(f'converting the output back to its original form')
        # create a label mapping
        label_mapping = {1:'Machine Failure', 0:'No Machine Failure'}
        y_pred_category = [label_mapping[pred] for pred in y_pred]
        
        return report, y_pred_category
    
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
        
    except EOFError as e:
        raise CustomException(e, sys)        