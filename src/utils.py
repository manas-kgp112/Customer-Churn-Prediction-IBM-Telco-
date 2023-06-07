# Importing standard libraries
import os
import sys
import numpy as np
import pandas as pd
import dill
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Importing sklearn_evaluation for pdf generation {for accuracy scores and classification report}
from sklearn_evaluation import plot


# GridSearchCV & RandomizedSearchCV import 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



# Importing Custom Exception and Logger
from src.exception import CustomException
from src.logger import logging



'''
    This script contains utilities for the project such as saving of the pkl files,
    evaluating models and loading model files.
'''

# This function saves the .pkl file at desired file
def save_object(file_path:str, obj):
    try:
        dirname = os.path.dirname(file_path)
        os.makedirs(dirname, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    

# This function evaluates models and tuning them.
def evaluate_models(X_train, Y_train, X_val, Y_val, models:dict, param_grid:dict):
    try:
        logging.info("Evaluating models.")
        # for storing performance analysis
        os.makedirs(os.path.join("artifacts", "performance"), exist_ok=True)
        os.makedirs(os.path.join("artifacts", "models"), exist_ok=True)
        # These dictionaries will store the classification report and accuracy stats for every model
        accuracies = {}

        for model_name, model in models.items():
            if model_name in param_grid:
                hyper_parameters = param_grid[model_name]
                logging.info(f"Implementing GridSearchCV() to find the version of {model_name}")
                grid_search = GridSearchCV(model, hyper_parameters, cv=5)
                grid_search.fit(X_train, Y_train)
                logging.info(f"Best version trained for {model_name}")
                best_model = grid_search.best_estimator_
                save_object(os.path.join("artifacts", "models", f"{model_name}.pkl"), best_model)
                Y_pred = best_model.predict(X_val)
                accuracy = accuracy_score(Y_val, Y_pred)
                logging.info(f"Printing classification reports for {model_name}")
                accuracies[model_name] = accuracy

                # Saving Confusion Matrix
                conf_mat = confusion_matrix(Y_val, Y_pred)
                cm_d = ConfusionMatrixDisplay(conf_mat)
                cm_d.plot(cmap='Blues')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.savefig(os.path.join("artifacts", "performance", f"ConfusionMatrix_{model_name}.pdf"), format='pdf', dpi=3000)
                logging.info(f"Confusion matrix saved for {model_name}")


                # Saving Classification Report
                plot.ClassificationReport.from_raw_data(Y_val, Y_pred)
                plt.savefig(os.path.join("artifacts", "performance", f"ClassificationReport_{model_name}.pdf"), format='pdf', dpi=3000)
                logging.info(f"Classification Report saved for {model_name}")
                


        logging.info("All models successfully evaluated and the best versions of each are selected.")



        # returning reports
        return (
            accuracies
        )
    except Exception as e:
        raise CustomException(e, sys)
    