# Importing libraries
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataframe = pd.DataFrame



# Importing Custom Exception and Logger
from src.exception import CustomException
from src.logger import logging



'''
    This script contains the data_vizualization code : loads some standard visualizations to "artifacts"
'''


# DataVisualizationConfig() class loads the dataset for further visualization processing
class DataVisualizationConfig:
    def __init__(self, load_path=os.path.join("artifacts", "plots")) -> None:
        self.load_path = load_path



class DataVisualization:
    # issues variable for visualization config
    def __init__(self) -> None:
        self.config_data_ = DataVisualizationConfig()


    # loads the categorical features in an array
    def get_categorical_features(self, data:dataframe):
        try:
            logging.info("Initiating data visulaization sequence...")
            logging.info("Extracting categorical features.")
            categorical_features = []
            for column in data:
                value_counts = pd.value_counts(data[column])
                value_counts_len = len(value_counts.index.to_list())

                if value_counts_len < 10 and value_counts_len!=1:
                    categorical_features.append(column)

            
            return (
                categorical_features
            )
        except Exception as e:
            raise CustomException(e, sys)
    

    # plots and loads the plots of categoical features as pdf
    def load_categorical_visualization(self, categorical_features, data):
        try:
            os.makedirs(self.config_data_.load_path, exist_ok=True)
            fig, axes = plt.subplots(9,2, figsize=(15,40))
            axes = axes.flatten()
            for i in range(len(categorical_features)):
                features = categorical_features[i]
                sns.countplot(x=features, data=data, palette = 'Set2', ax=axes[i], hue='Churn Value')

            plt.tight_layout()
            plt.savefig(os.path.join(self.config_data_.load_path, "categorical_features_plot.pdf"), dpi=3000, format='pdf')
            logging.info(f"Plots saved as : {os.path.join(self.config_data_.load_path, 'categorical_features_plot.pdf')}")
        except Exception as e:
            raise CustomException(e, sys)
        

    # plots and loads the plots of continous features as pdf
    def load_continous_visualization(self, data_services):
        try:
            continous_data = ['Tenure in Months', 'Total Revenue', 'Total Charges']
            fig, axes = plt.subplots(3,1, figsize=(10,15))
            axes = axes.flatten()
            for i in range(len(continous_data)):
                features = continous_data[i]
                sns.histplot(x=features, data=data_services, ax=axes[i])
            plt.tight_layout()
            plt.savefig(os.path.join(self.config_data_.load_path, "continous_features_plot.pdf"), dpi=3000, format='pdf')
            logging.info(f"Plots saved as : {os.path.join(self.config_data_.load_path, 'continous_features_plot.pdf')}")
        except Exception as e:
            raise CustomException(e, sys)
        