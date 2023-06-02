# Importing libraries
import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



# Importing Custom Exception and Logger
from src.exception import CustomException
from src.logger import logging



'''
    This script contains the data_transformation pipeline : performs preprocessing on the data.
    Steps done in this script include :
        1}
        2}
        3}

'''


# specifies path to store preprocessor model file
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("atifacts", "preprocessor.pkl")



# DataTransformation() class processes the data
class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformation()

    # This function exports the data transformation object
    def data_transformation_object(self):
        try:
            categorical_features = [
            'Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Multiple Lines',
            'Internet Service', 'Online Security', 'Online Backup', 'Device Protection', 'Tech Support',
            'Streaming TV', 'Streaming Movies', 'Contract','Paperless Billing', 'Payment Method', 'Churn Value'
        ]
            cat_pipeline = Pipeline(
                steps = [
                    ("label encoder", LabelEncoder())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("categorical_pipeline", cat_pipeline, categorical_features)
                ]
            )

            return (
                preprocessor
            )
        except Exception as e:
            raise CustomException(e, sys)