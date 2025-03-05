import os
import logging

def error_message(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{}], \
                    line number [{}], error_message [{}]".format(
                        file_name, exc_tb.tb_lineno, str(error)
                    )
                    
    return error_message

    
    