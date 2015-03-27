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
#from build import BuildEventPlanner

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
    tests = [[-77.003023,  38.907024,  0,  0,  0,  0.304347826,  0.196754564,  0.17693837,  7.8015,  4.844,  0.54332312,  1,  0,  0], #Robbery
             [-76.94675,  38.899199,  0,  0,  0,  0.215223097,  0.541899441,  0.425925926,  2.6393,  2.458,  0.247386337,  0,  1,  0], #Assault
             [-77.018179,  38.976086,  0,  0,  0,  0.083538084,  0.584830339,  0.341317365,  7.038,  4.128,  1.71944751,  1,  0,  0], #Robbery
             [-77.002205,  38.951855,  0,  0,  0,  0.209090909,  0.602040816,  0.348684211,  7.0375,  3.307,  0.768374939,  0,  0,  1], #Burglary
             [-76.935259,  38.908186,  0,  0,  0,  0.141304348,  0.620689655,  0.581280788,  2.725,  2.731,  0.758946655,  0,  0,  1], #Burglary
             [-77.021917,  38.906445,  0,  0,  0,  0.124314442,  0.363977486,  0.177743431,  8.558,  6.981,  0.349857117,  0,  1,  0]] #Sex Abuse
             
    for test in tests:
        print classifier.classify(test)
    
