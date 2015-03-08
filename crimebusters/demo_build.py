# -*- coding: utf-8 -*-
"""
Created on Tue., Feb. 24, 2015

@author: kathleen_llontop
"""
########################################################################
import os
import csv
import sys 
import time
import json
import pickle 
import random
from sklearn import svm
from datetime import datetime

########################################################################
class BuildEventPlanner(object):
    	
    def __init__(self, in_data, **kwargs):
        
        #today's date        
        today = datetime.now().strftime('%Y-%d-%m')
        
        self.filename = in_data
        self.features = None
        self.classification_field = kwargs.pop("classification_field", "OFFENSE")
        self.accuracy = None
        
        #output directories
        self.out_model = kwargs.pop("out_model",os.path.join("..","fixtures", "model_{0}.pickle".format(today)))
        self.out_model_log = kwargs.pop("out_log",os.path.join("..","fixtures", "info_{0}.json".format(today)))
        
        #check the output paths when initialized
        self.__check_output_paths()

        #Record processing time
        self.train_time = None
        self.feature_time = None    
        self.test_time = None
        self.build_time = None
        
        
        #test size for validation. 10 for 10%
        self.test_size = kwargs.pop("test_size",10)
        
        #run cross validation. Defaults to True at the start
        self.validate = kwargs.pop("validate", True)
    		
    def featureset(self):
        """
        load data into the event planner. There are no inputs becase they are 
        grabbed from the object properties.
        
        These are the fields I am working with:
        [OFFENSE	MIDNIGHT	DAY	EVENING	VacantUnitRate	LongCommuteRate	PublicTransitCommuters	MedIncomeInTenThousands	MedHomeValueInHundredThousands	DistFromMetKm]

        
        I removed the extraneous data and stored this as a separate csv. I 
        was not able to use a numpy array b/c they do not support multiple 
        data types. I am not using pandas either b/c Allen and Ben 
        keep saying that it is not for production code. 
        """
        if self.features is None:
            #rb means read binary 
            start = time.time()
            self.features = []
            
            #open the files to read the data into the object
            with open(self.filename, "rb") as in_data:
                csv_reader = csv.DictReader(in_data)
                for line in csv_reader:
                    
                    #removes the criminal offense into its own list
                    crime = line.pop(self.classification_field)
                    
                    #all the other field values are extracted here.
                    #using list comprehension to ensure all the features
                    #fields are in the same order
                    features = [line[field_name] for field_name in line.keys()]
                    
                    self.features.append((crime,features))
        
            self.feature_time = time.time() - start
        
        return self.features
        
    def _parse_X_and_y(self, features):
        """
        Used to parse out X and y because they are in a tuple.
        
        Returns X and y in that order
        """
        #X's the support vectors. These all have to be 
        #numeric values so additional data transformation may be required. 
        #Example error from the initial test ValueError: could not convert string to float: MIDNIGHT
        X = []
        
        #y is the result/classificaiton. Can be string or integer
        y = []
        
        for item in features:
            offense, fields = item
            y.append(offense)
            X.append(fields)
            
        return X, y
                
        	
    def train(self, featureset=None, probability=True):
        """
        This is where the algorithmn will be trained using
        an input featureset. The data should come from
        self.featureset() but the input parameter 
        allows for reuse.
        
        Pass in featureset during cross validation 
        """
        try:
            sys.stdout.write("Training...\n")
            sys.stdout.write(time.ctime()+"\n")            
            
            start = time.time()
            
            featureset = featureset or self.featureset()
            
            X, y = self._parse_X_and_y(featureset)
            
            #Uses all defaults including an radial basis function (rbf)
            #kernal. Refer to this site: http://scikit-learn.org/stable/modules/svm.html
            #Would also be worthwile to investigate sklearn.svm.LinearSVC and sklearn.linear_model.SGDClassifier
            
            classifier = svm.SVC(probability=probability)
            
            classifier.fit(X, y)
            
            #This is here to preserve the train_time from the initial 
            #classifier build. If a user decides to cross_validate then this 
            #could get overwritten            
            if not self.train_time:            
                self.train_time = time.time() - start 
            
            return classifier
        
        except ValueError as error:
            #SVM will throw this if all the support vectors are not numeric
            raise ValueError(error)
        
    
    def build(self):
        """
        This is where the class will build the output and write
        to disk likely using pickle. 
        """
        sys.stdout.write("Building classifier...\n")
        sys.stdout.write(time.ctime()+"\n")
        
        start = time.time()
        
        #Builds the classifier with all of the data
        classifier = self.train()
        
        with open(self.out_model, "w") as target:
            pickle.dump(classifier, target, pickle.HIGHEST_PROTOCOL)
            
        if self.validate:
            self.cross_validate()
            
        self.build_time = time.time() - start
        self.write_details()
        
                
        
    def cross_validate(self):
        """
        A function to test the accuracy of the 
        model.
        
        See documentation here: http://scikit-learn.org/stable/modules/cross_validation.html
        """
        
        sys.stdout.write("Cross validation...\n")
        sys.stdout.write(time.ctime()+"\n")        
        
        start = time.time()
        
        #not sure if i can use self.features or should generate a whole new featureset
        if not self.features:
            features = self.featureset()
        else:
            features = self.features
        
        #get 10% sample size
        test_size = len(features)/10
        
        #randomize feature organization
        random.shuffle(features)
        
        #train 90%, test 10%
        train = features[test_size:]
        test = features[:test_size]
        
        classifier = self.train(featureset=train, probability=False)
        self.accuracy = self.get_score(classifier, test)
        self.test_time = time.time() - start 
        
        sys.stdout.write("Model accuracy: {0}%\n".format(round(self.accuracy*100,3)))
        
    def get_score(self, classifier, test_set):
        """
        Returns the score of the classifier. test_set needs
        to be parsed because y and X are in a tuple (y, X) per self.featureset()
        """
        X, y = self._parse_X_and_y(test_set)
        
        return classifier.score(X,y)
        
    def __check_output_paths(self):
        """
        function that returns where the model outputs
        are written
        """
        out_files = [self.out_model,self.out_model_log]
        for path in out_files:
            if os.path.exists(path):
                raise Exception("Can't overwrite file at {0}!".format(path))
    
    def write_details(self):
        """
        Function to write model stats such as date and runtime.
        """
        details = {
        "in data" : self.filename,
        "classification field" : self.classification_field,
        "accuracy" : self.accuracy,
        "train time" : self.train_time,
        "build time" : self.build_time,
        "test time" : self.test_time,
        "feature strucutre time" : self.feature_time,
        "validation" : self.validate,
        "test size" : "{0}%".format(self.test_size) if self.test_size else self.test_size
        }
        
        with open(self.out_model_log, 'w') as target:
            json.dump(details, target, indent=4)
        
    
if __name__ == "__main__":
    infile = os.path.join('..',r'Data/Crime_for_classifier_Normalized_Nulls_Removed.csv')
    event_planner = BuildEventPlanner(infile)
    event_planner.build()
    #raw_input("\nPress enter to quit.")
        
		