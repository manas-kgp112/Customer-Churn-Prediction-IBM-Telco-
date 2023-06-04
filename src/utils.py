# Importing standard libraries
import os
import sys
import numpy as np
import pandas as pd
import dill



# Importing Custom Exception and Logger
from src.exception import CustomException
from src.logger import logging



'''
    This script contains utilities for the project such as saving of the pkl files.
'''

# This function saves the .pkl file at desired file
def save_object(file_path, obj):
    try:
        dirname = os.path.dirname(file_path)
        os.makedirs(dirname, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)