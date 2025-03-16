import os
import sys

import pandas as pd
import numpy as np 
import pickle
import dill


from sklearn.model_selection import StratifiedKFold, GridSearchCV
from src.exception import CustomException
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score


def save_object(file_path, obj):
    
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
