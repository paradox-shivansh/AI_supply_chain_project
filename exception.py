import sys
from typing import Optional


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """Extract detailed error information including file name and line number."""
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{line_number}] "
        f"error message [{str(error)}]"
    )
    return error_message


class CustomException(Exception):
    """
    Custom exception class for Amazon Supply Chain Intelligence.
    Captures file name, line number, and error message for better debugging.
    """

    def __init__(self, error_message: Exception, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message
