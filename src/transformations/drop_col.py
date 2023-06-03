# Importing libs
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# Importing Custom Exception and Logger
from src.exception import CustomException
from src.logger import logging



'''
    This script drops specific cols as given by user.
    It is specifically created to be used in the pipline directly.
'''


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
        self.columns = columns

    # def fit(self, X, y=None):
    #     return self

    # def transform(self, X):
    #     return X.drop(self.columns, axis=1)

    def print(self):
        return (self.columns)