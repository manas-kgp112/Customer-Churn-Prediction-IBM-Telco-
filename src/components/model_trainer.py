# Importing libraries
import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass


# Importing models to train
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    StackingClassifier,
    VotingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# Importing performance metrics analysis libraries
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
    precision_recall_curve
)


# Importing Custom Exception and Logger
from src.exception import CustomException
from src.logger import logging


# Importing custom modules
from src.utils import (
    save_object, # saves the model on the specified path
    evaluate_models # evaluate models and tunes them {export best scoring model}
)


'''
    This script contains all the models that will be trained on the dataset. It will also
    evaluate performance metrics of all the models and help them compare against each other.
    We could also export these metrics and export them.

'''



# specifies the path for the trained model to be stored
@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join("artifacts", "model", "model.pkl")




# The ModelTrainer() class configures, tunes, traines and export models to specified paths for further use. This also evaluates performance of the models based on few performance metrics
class ModelTrainer:

    # This function initialises the path where the fint-tuned/trained model must be stored. (extra configuration may also be added in the config_class).
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_data, val_data):
        try:
            logging.info("Splitting train/val datasets into X and Y inputs.")
            X_train = train_data.drop(['Churn Value'], axis=1)
            Y_train = train_data['Churn Value']
            X_val = val_data.drop(['Churn Value'], axis=1)
            Y_val = val_data['Churn Value']

            logging.info("Initialising model building process...")

            # Creating a dict for all the models that will be trained on the dataset
            models = {
                'Logistic Regression' : LogisticRegression(),
                'Gaussian Naive Bayes' : GaussianNB(),
                'K Nearest Neighbors' : KNeighborsClassifier(),
                'Support Vector Machine' : SVC(probability=True),
                'Decision Tree Classifier' : DecisionTreeClassifier(),
                'Random Forest Classifier' : RandomForestClassifier(),
                'Bagging Classifier' : BaggingClassifier(
                    base_estimator=RandomForestClassifier()
                ),
                'Gradient Boosting Classifier' : GradientBoostingClassifier(),
                'AdaBoost' : AdaBoostClassifier(
                    base_estimator=DecisionTreeClassifier()
                ),
                'Stacking Classifier' : StackingClassifier(
                    estimators=[
                        ('log_reg', LogisticRegression()),
                        ('random_forest', RandomForestClassifier()),
                        ('grad_boost', GradientBoostingClassifier())
                    ]
                ),
                'Voting Classifier' : VotingClassifier(
                    estimators=[
                        ('log_reg', LogisticRegression()),
                        ('random_forest', RandomForestClassifier()),
                        ('grad_boost', GradientBoostingClassifier())
                    ]
                ),
                'XgBoost' : XGBClassifier(),
                'LightGBM' : LGBMClassifier(
                    scale_pos_weight =3,
                    random_state=42,
                    objective = 'binary'
                ),
                'CatBoost' : CatBoostClassifier()
            }



            # Dictionary containing all the parameters for all models
            # Hyperparameter Tuning
            params = {
                'Logistic Regression' : {
                    'penalty' : ['l2', 'elasticnet'],
                    'C' : [0.1, 1, 10],
                    'max_iter' : [100, 1000, 10000]
                },
                'Gaussian Naive Bayes' : {}, # No hyperparameters to tune
                'K Nearest Neighbors' : {
                    'n_neighbors' : [1, 5, 10],
                    'weights' : ['uniform', 'distance']
                },
                'Support Vector Machine' : {
                    'C' : [0.1, 1],
                    'kernel' : ['rbf'],
                    'gamma' : [0.1, 0.01]
                },
                'Decision Tree Classifier' : {
                    'criterion' : ['gini', 'log_loss'],
                    'max_depth' : [None, 10, 100]
                },
                'Random Forest Classifier' : {
                    'criterion' : ['gini', 'log_loss'],
                    'max_depth' : [None, 10, 100],
                    'n_estimators' : [100, 200, 300]
                },
                'Bagging Classifier' : {
                    'n_estimators' : [10, 20, 30, 100]
                },
                'Gradient Boosting Classifier' : {
                    'learning_rate' : [0.1, 1],
                    'n_estimators' : [100, 200, 300]
                },
                'AdaBoost' : {
                    'n_estimators' : [50, 100, 200, 300]
                },
                'Stacking Classifier' : {}, # No hyperparameter to tune
                'Voting Classifier' : {}, # No hyperparameter to tune
                'XgBoost' : {},
                'LightGBM' : {
                    'learning_rate' : [0.1, 0.01],
                    'max_depth' : [-5, -10, -20] 
                },
                'CatBoost' : {
                    'depth' : [6, 8, 10],
                    'learning_rate' : [0.01, 0.05, 0.1],
                    'iterations' : [30, 50, 100]
                }

            }




            # evalutaing the models and using GridSearchCV() to tune them and exporting the best performing model.
            classification_reports, accuracies = evaluate_models(X_train, Y_train, X_val, Y_val, models, params)


            return (
                classification_reports,
                accuracies
            )


        except Exception as e:
            raise CustomException(e, sys)