# Importing libraries
import os
import sys
import pandas as pd
import numpy as np



# Importing Custom Exception and Logger
from src.exception import CustomException
from src.logger import logging


# Importing custom modules for vizualization, model-training etc.
from src.components.data_visualization import DataVisualization
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer



'''
    This script contains the data_ingestion pipeline : loads the data from the "artifacts"

'''



# DataIngestionConfig() class loads the path all of the datasets available and stores the additional configuration information
class DataIngestionConfig:
    def __init__(self,
                 telco_customer_churn_path:str = os.path.join("artifacts", "data", "Telco_customer_churn.xlsx"),
                 telco_customer_churn_demographics_path:str = os.path.join("artifacts", "data", "Telco_customer_churn_demographics.xlsx"),
                 telco_customer_churn_location_path:str = os.path.join("artifacts", "data", "Telco_customer_churn_location.xlsx"),
                 telco_customer_churn_services_path:str = os.path.join("artifacts", "data", "Telco_customer_churn_services.xlsx"),
                 telco_customer_churn_population_path:str = os.path.join("artifacts", "data", "Telco_customer_churn_population.xlsx"),
                 telco_customer_churn_status_path:str = os.path.join("artifacts", "data", "Telco_customer_churn_status.xlsx")) -> None:
        
        self.churn_path  = telco_customer_churn_path
        self.churn_demographics_path = telco_customer_churn_demographics_path
        self.churn_location_path = telco_customer_churn_location_path
        self.churn_services_path = telco_customer_churn_services_path
        self.churn_population_path = telco_customer_churn_population_path
        self.churn_status_path = telco_customer_churn_status_path
        



# DataIngestion() class loads the data and generates the pipeline for data pre-processing upto model training and feature-importance ranking
class DataIngestion:
    # stores the ingestionconfig as variable named "data_ingestion_config"
    def __init__(self) -> None:
        self.data_ingestion_config = DataIngestionConfig()

    
    def initiate_data_ingestion(self):
        logging.info("Initiating Data Ingestion sequence...")
        try:
            count = 0
            churn_df = pd.read_excel(self.data_ingestion_config.churn_path)
            count += 1
            churn_location_df = pd.read_excel(self.data_ingestion_config.churn_location_path)
            count += 1
            churn_population_df = pd.read_excel(self.data_ingestion_config.churn_population_path)
            count += 1
            churn_services_df = pd.read_excel(self.data_ingestion_config.churn_services_path)
            count += 1
            churn_status_df = pd.read_excel(self.data_ingestion_config.churn_status_path)
            count += 1
            churn_df_demographics = pd.read_excel(self.data_ingestion_config.churn_demographics_path)
            count += 1
            logging.info(f"Extracted {count} files as DataFrames.")


            return (
                churn_df,
                churn_services_df,
                self.data_ingestion_config.churn_path
            )
        except Exception as e:
            raise CustomException(e, sys)







if __name__ == "__main__":
    # data ingestion sequence initiation
    # data_ingestion = DataIngestion()
    # data, data_services, data_path = data_ingestion.initiate_data_ingestion()


    # # data visulaization sequence initiation
    data_visualization = DataVisualization()
    categorical_features = data_visualization.get_categorical_features(data=data)
    data_visualization.load_categorical_visualization(categorical_features=categorical_features, data=data)
    data_visualization.load_continous_visualization(data_services=data_services)


    # data transformation sequence initiation
    data_transformation = DataTransformation()
    # data_processor, drop_features, dummy_features = data_transformation.data_transformation_object()
    transformed_data = data_transformation.initiate_data_transformation(data=data)
    _, _, _, _, train_data, test_data = data_transformation.split_dataset(transformed_data=transformed_data)



    # model training sequence initiation
    model_trainer = ModelTrainer()
    accuracy_reports = model_trainer.initiate_model_trainer(train_data=train_data, val_data=test_data)