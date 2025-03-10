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


