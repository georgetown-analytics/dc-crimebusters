# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 21:11:35 2015

@author: andrew_woizesko
"""

#import pickle
#import time
#from datetime import datetime
import csv

class BuildEventPlanner(object):
    """
    Creates a classifier that will be used to classify a user's
    travel plans. The classifier algorithmn has yet to be 
    identified but possible options are logistic regression (logit/maxent)
    and a support vector machine (SVM)
    """
    
    def __init__(self, in_file):
        """
        This is where the class will be initialized. Parameters
        will be dictated by other class functions.
        """
        self.in_file = in_file
        self.features = None
    
    def featureset(self):
        """
        This is where the training features will be loaded. 
        This will likely be the crime incidents that contain 
        the census data and metro data. The featureset data should
        be stored in the fixtures folder
        """
        with open(self.in_file, 'rb') as in_data:
            pass
    
    def train(self, featureset=None):
        """
        This is where the algorithmn will be trained using
        an input featureset. The data should come from
        self.featureset() but the input parameter 
        allows for reuse.
        """
        featureset = featureset or self.featureset()
        pass
    
    def build(self):
        """
        This is where the class will build the output and write
        to disk likely using pickle. 
        """
        pass
    
    def cross_validate(self):
        """
        A function to test the accuracy of the 
        model.
        """
        pass
    
    def get_output_paths(self):
        """
        function that returns where the model outputs
        are written
        """
        pass
    
    def write_details(self):
        """
        Function to write model stats such as date and runtime.
        """
        pass
    
if __name__ == "__main__":
    print "You need to code first!"
    raw_input("\nPress enter to quit.")
        