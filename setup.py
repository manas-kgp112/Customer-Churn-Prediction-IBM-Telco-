# Importing libraries
import os
from setuptools import find_packages, setup
from typing import List


'''
    This file (setup.py) contains the python code to execute the installation of packages for the project
    and initializing the setup configuration
'''

# get_requirements() returns all the packages required for the project
def get_requirements(file_path:str) -> List[str]:
    with open(file_path, "r") as f:
        reqs = f.readlines()
        reqs = [req.replace("\n","") for req in reqs]

        if "-e ." in reqs:
            reqs.remove("-e .")

        return reqs
    
# Utility function to read the README file.
def get_readme(file_path:str) -> str:
    return open(os.path.join(os.path.dirname(__file__), file_path)).read()


# Project Configuration
# reference link : https://pythonhosted.org/an_example_pypi_project/setuptools.html
setup (
    name = "customer_churn_prediction",
    version = "0.0.1",
    author = "Manas Sharma",
    author_email = "manassharma.ms2593@gmail.com",
    description = ("Customer Churn Prediction system trained on IBM's [Telco churn dataset]"),
    packages = find_packages(),
    long_description = get_readme('README.md'),
    install_requires = get_requirements('requirements.txt')
)