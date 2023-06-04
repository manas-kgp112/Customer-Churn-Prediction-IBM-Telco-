# Importing libraries
import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split



# Importing Custom Exception and Logger
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# Importing custom modules



'''
    This script contains the data_transformation pipeline : performs preprocessing on the data.
    Steps done in this script include :
        1} Label Encoding for categorical data
        2} Dummy variables
        3}

'''


# specifies path to store preprocessor model file
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")



# DataTransformation() class processes the data
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # This function exports the data transformation features
    def data_transformation_object(self):
        try:
            logging.info("Collecting features.")
            categorical_features = [
                'Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Multiple Lines',
                'Internet Service', 'Online Security', 'Online Backup', 'Device Protection', 'Tech Support',
                'Streaming TV', 'Streaming Movies', 'Contract','Paperless Billing', 'Payment Method', 'Churn Value'
            ]

            dummy_cat_features = [
                'Gender', 'Senior Citizen', 'Partner', 'Dependents', 'Phone Service', 'Multiple Lines',
                'Internet Service','Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 
                'Streaming TV', 'Streaming Movies', 'Contract','Paperless Billing', 'Payment Method'
            ]

            drop_features = [
                'CustomerID', 'Lat Long', 'Churn Reason', 'Country', 'State',
                'City', 'Zip Code', 'Churn Label', 'Count'
            ]
            
            logging.info("Data transformation object extracted.")


            return (
                categorical_features,
                drop_features,
                dummy_cat_features
            )
        except Exception as e:
            raise CustomException(e, sys)
        



    # This function collects the data_transformation object() and applies it on the dataset
    def initiate_data_transformation(self, data):
        try:
            logging.info("Initiating data transformation sequence...")
            categorical_features, drop_features, dummy_features = self.data_transformation_object()

            # Dumping some features and changing dtypes
            data_new = data.drop(drop_features, axis=1)
            data_new['Total Charges'] = pd.to_numeric(data_new['Total Charges'], errors='coerce')

            # Dumping all null values
            data_new.dropna(axis=0, inplace=True)
            logging.info("Applying preprocessing object on training dataframe")

            label_enc = LabelEncoder()
            for col in categorical_features:
                data_new[col] = label_enc.fit_transform(data_new[col])
            logging.info("Data transformed successfully")

            # Creating dummy variables
            data_new_f = pd.get_dummies(data_new, columns=dummy_features, drop_first=True)

            # saving the datafile as csv
            data_new_f.to_excel(os.path.join("artifacts", "data", "transformed_churn.xlsx"))
            logging.info("Transformed data saved as .xlsx @ [artifacts/data/transformed_churn.xlsx]")

            return (
                data_new_f
            )

        except Exception as e:
            raise CustomException(e, sys)




    # This function splits the data into train and validation set
    def split_dataset(self, transformed_data):
        try:
            X = transformed_data.drop(['Churn Value'], axis=1)
            Y = transformed_data['Churn Value']

            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
            train_data, val_data = train_test_split(transformed_data, test_size=0.2, random_state=42)

            return (
                X_train,
                X_val,
                Y_train,
                Y_val,
                train_data,
                val_data
            )
        except Exception as e:
            raise CustomException(e, sys)