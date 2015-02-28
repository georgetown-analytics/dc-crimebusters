# -*- coding: utf-8 -*-
"""
Created on Tue., Feb. 24, 2015

@author: kathleen_llontop
"""
########################################################################
import pandas as pd
import numpy as np
from sklearn import cross_validation

########################################################################
class BuildEventPlanner(object):

	filename = 'CrimeEventsWCensusMetDistAndName.csv'
	
	def __init__(self, filename):
		self.filename = filename
		
	def featureset(self, filename):
		# Produce the test and training sets
		# Read csv file into a structured pandas dataframe
		df = pd.read_csv(filename, sep = ",", quotechar = '"',skipinitialspace = True)

		# For the purpose of using scikit learn functions on this data,
		# tranform dataframe into a numpy array
		df_array = np.array(df)

		## Split data into test and train sets using KFold
		# We can decide on the n and k
		# Setting random_state=0 sets the seed so the splits are reproducible
		# Not sure if we want shuffle = True before splitting
		kf = KFold(n, k, indices = True, random_state=0)
		
		# kf now contains a train and test set which can be viewed by running the following code:
#		for train_index, test_index in kf:
#...			print("TRAIN: %s TEST: %s" % (train_index, test_index))
#...    		X_train, X_test = X[train_index], X[test_index]
#...    		y_train, y_test = y[train_index], y[test_index]
	
	def train(self, featureset=None):
        """
        This is where the algorithmn will be trained using
        an input featureset. The data should come from
        self.featureset() but the input parameter 
        allows for reuse.
        """
		# Input feature set should be X_train, y_train from above
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
		# A good function for testing the accuracy of a model
		# appears to be cross_validation.cross_val_score.
		# This is an example of how to use it below:
		clf = svm.SVC(kernel='linear', C=1)
		scores = cross_validation.cross_val_score(
 ...    		clf, X_train, y_train, cv=k) # where k = number of folds

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
        
		