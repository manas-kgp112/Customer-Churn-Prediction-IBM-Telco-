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

# Importing ReportLab for pdf generation {for accuracy scores and classification report}
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter


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
        content = []
        os.makedirs(os.path.join("artifacts", "performance"), exist_ok=True)
        # Define a style for the paragraphs
        styles = getSampleStyleSheet()
        # These dictionaries will store the classification report and accuracy stats for every model
        classification_reports = {}
        accuracies = {}

        for model_name, model in models.items():
            if model_name in param_grid:
                content.append(Paragraph(model_name, style=styles['Title']))
                hyper_parameters = param_grid[model_name]
                logging.info(f"Implementing GridSearchCV() to find the version of {model_name}")
                grid_search = GridSearchCV(model, hyper_parameters, cv=5)
                grid_search.fit(X_train, Y_train)
                best_model = grid_search.best_estimator_
                Y_pred = best_model.predict(X_val)
                accuracy = accuracy_score(Y_val, Y_pred)
                content.append(Paragraph(f"Accuracy Score : {accuracy_score}", style=styles['Normal']))
                class_report = classification_report(Y_val, Y_pred)
                content.append(Paragraph(class_report, style=styles['Normal']))
                content.append(Paragraph("<br/><br/>", styles["Normal"])) # for gap between two models
                logging.info(f"Printing classification reports for {model_name}")
                accuracies[model_name] = accuracy
                classification_reports[model_name] = class_report

                # Saving Confusion Matrix
                conf_mat = confusion_matrix(Y_val, Y_pred)
                cm_d = ConfusionMatrixDisplay(conf_mat)
                cm_d.plot(cmap='Blues')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                plt.savefig(os.path.join("artifacts", "performance", f"ConfusionMatrix_{model_name}.pdf"), format='pdf', dpi=3000)


        # Printing classification reports and accuracy scores
        performance_doc = SimpleDocTemplate(os.path.join("artifacts", "performance", "accuracy.pdf"), pagesize=letter)
        performance_doc.build(content)

        logging.info("All models successfully evaluated and the best versions of each are selected.")



        # returning reports
        return (
            classification_reports,
            accuracies
        )
    except Exception as e:
        raise CustomException(e, sys)