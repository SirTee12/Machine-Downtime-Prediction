import os
import sys
from src.exception import CustomException
from src.logging import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig

@dataclass
class dataIngestionConfig:
    train_data_path: str = os.path.join('Artifacts', 'train.csv')
    test_data_path: str = os.path.join('Artifacts', 'test.csv')
    raw_data_path: str = os.path.join('Artifacts', 'raw.csv')
    

        