# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 21:31:48 2015

@author: andrew_woizesko
"""
###########################################################################
## Imports
###########################################################################

import pickle

from conf import settings
from build import BuildEventPlanner

###########################################################################
## Event Classifier
###########################################################################

class EventClassifier(object):
    """
    Class that will be used to classify user input
    events. The event classifier will use the model 
    from build.py
    """
    
    def __init__(self, model=None):
        """
        Initialize the event classifier with the model
        created in build.py
        """
        
		## Get the default model from the settings if it isn't passed in
		model = model or settings.model
		
		## Load the model from the pickle
		with open(model, 'rb') as pkl:
			self._classifier = pickle.load(pkl)
			
    
    def classify(self, instance):
        """
        This is the function that will take the user 
        input (instance) and return the probability that the 
        user will encounter a crime on their trip
        """
        		
		## Use the classifier to predict the probabilities of each crime
		event_prob = self._classifier.predict_proba(instance)
		
		return event_prob
    
    def explain(self):
        """
        Not sure if we'll need this.
        """
		## Don't think we need this
        pass
    

if __name__ == "__main__":
    classifier = EventClassifier()
	