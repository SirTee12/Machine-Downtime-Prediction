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
    recall_score,
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
    