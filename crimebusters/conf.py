# -*- coding: utf-8 -*-
"""
Created on Thu Mar 05 13:25:56 2015

@author: andrew_woizesko
"""

import os 
import confire

class CrimeBustersConfig(confire.Configuration):
    """
    configuration for the application. See documentation
    here: http://pythonhosted.org/confire/
    
    This reads through a .yaml file in a conf directory (not in online repo)
    and pulls out the file paths. This is a good way to keep personal 
    file directories and other personal applicaiton information
    off the internet and provides a single location to store all application
    settings. 
    
    "data" is the input data for the classifier
    "shapefile" is the input shapefile for spatial regression
    "spatial_regression_summary" is the output file for spatial regression
    dbf_file is the file associated with the shapefile. 
    """
    CONF_PATHS = [
    "/etc/path_variables.yaml", 
    os.path.expanduser('~/.path_variables.yaml'),
    os.path.abspath(os.path.join("..","conf/path_variables.yaml"))
    ]
    
    data = None
    shapefile = None
    spatial_regression_summary = None
    dbf_file = None
    crime_data = None
    
settings = CrimeBustersConfig.load()

if __name__ == "__main__":
    print settings