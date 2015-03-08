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
    """
    CONF_PATHS = [
    "/etc/path_variables.yaml", 
    os.path.expanduser('~/.path_variables.yaml'),
    os.path.abspath(os.path.join("..","conf/path_variables.yaml"))
    ]
    
    data = None
    
settings = CrimeBustersConfig.load()

if __name__ == "__main__":
    print settings