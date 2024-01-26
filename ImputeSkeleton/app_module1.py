import logging
import os

def appfunction_of_module1(loglevel, progress_bar,resources,args):
    logging.info('[INFO] Welcome in ImputeSkeleton')
    logging.debug('[DEBUG] Now in appfunction_of_module1() : {}'.format(appfunction_of_module1))
    logging.debug('[DEBUG] Parsed arguments = {}'.format(args))
    logging.debug('[DEBUG] loglevel : {}, progress_bar : {}, resources located in {}'.format(loglevel,progress_bar,resources))
    return
