# Importing libraries
import sys
from src.logger import logging

'''
    This script raises CustomException and generates traceback calls.
    {custom error message}
'''


class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        self.error_message = f"Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
            file_name, exc_tb.tb_lineno, str(error_message)
        )

    def __str__(self) -> str:
        return self.error_message