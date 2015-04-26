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
import dill 
import numpy as np
from sklearn import svm
from conf import settings
from datetime import datetime
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.grid_search import GridSearchCV

########################################################################
class BuildEventPlanner(object):
    	
    def __init__(self, in_data, **kwargs):
        
        #today's date        
        today = datetime.now().strftime('%Y-%d-%m')
        
        self.filename = in_data or settings["data"]
        self.features = None
        self.labels = None
        self.classification_field = kwargs.pop("classification_field", "OFFENSE")
        self.accuracy = None
        self.null_value = kwargs.pop("null_value", -999)
        
        self.params = kwargs.pop('params', {'C':[.1, 1, 10, 100]})
        
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
        self.n_folds = kwargs.pop("n_folds",12)
        
        #run cross validation. Defaults to True at the start
        self.validate = kwargs.pop("validate", True)
    		
    def load_data(self):
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
        if (self.features is None) or (self.labels is None):
            #rb means read binary 
            start = time.time()
            support_features = []
            target_labels = []
            
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
                    
                    support_features.append(features)
                    target_labels.append(crime)
        
            self.feature_time = time.time() - start
        
        self.features = np.array(support_features)
        self.labels = np.array(target_labels)
        
        return self.features, self.labels
        	
    def train(self, X=None,y=None):
        """
        This is where the algorithmn will be trained using
        an input featureset. The data should come from
        self.load_data() but the input parameter 
        allows for reuse.
        
        Pass in featureset during cross validation 
        """
        try:
            sys.stdout.write("Training...\n")
            sys.stdout.write(time.ctime()+"\n")            
            
            start = time.time()
            
            if (X is None) or (y is None):
                if (self.features is None) or (self.labels is None):
                    X, y = self.load_data()
                else:
                    X = self.features
                    y = self.labels
                
            #fill in missing values
            imputer = Imputer(missing_values=self.null_value, strategy='mean')

            #Uses all defaults including an radial basis function (rbf)
            #kernal. Refer to this site: http://scikit-learn.org/stable/modules/svm.html
            #Would also be worthwile to investigate sklearn.svm.LinearSVC and sklearn.linear_model.SGDClassifier
            
            
            #Tried class_weight="auto" but it produced a worse fit 38%
            clf = svm.SVC() #currently at 66%
            
            grid_search = GridSearchCV(clf, param_grid=self.params)
            
            #classifier = DecisionTreeClassifier()  #produced 30% accuracy    
            classifier = Pipeline([('imputer', imputer), ('clf', grid_search)])
            
            classifier.fit(X, y)
            
            #This is here to preserve the train_time from the initial 
            #classifier build. If a user decides to cross_validate then this 
            #could get overwritten            
            if self.train_time is None:            
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
        
        with open(self.out_model, "wb") as target:
            sys.stdout.write("Saving model to {0}\n".format(self.out_model))
            dill.dump(classifier, target, dill.HIGHEST_PROTOCOL)
            
        if self.validate:
            self.cross_validate()
            
        self.build_time = time.time() - start
        self.write_details()
        return classifier
        
    def cross_validate(self):
        """
        A function to test the accuracy of the 
        model.
        
        See documentation here: http://scikit-learn.org/stable/modules/cross_validation.html
        """
        sys.stdout.write("Cross validation...\n")
        sys.stdout.write(time.ctime()+"\n")        
        
        start = time.time()
        
        n_folds = self.n_folds
        scores = []
        
        #not sure if i can use self.features or should generate a whole new featureset
        if (self.features is None) or (self.labels is None):
            features, labels  = self.load_data()
        else:
            features = self.features
            labels = self.labels
        
        kf = cross_validation.KFold(len(features), n_folds, random_state=np.random)
        
        for train, test in kf:
            X_train, X_test = features[train], features[test]
            y_train, y_test = labels[train], labels[test]
        
        
            classifier = self.train(X_train, y_train)
            scores.append(classifier.score(X_test, y_test))
        self.accuracy = sum(scores)/len(scores) #average score of the folds
        self.test_time = time.time() - start 
        
        sys.stdout.write("Model accuracy: {0}%\n".format(round(self.accuracy*100,3)))

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
        "folds" : self.n_folds
        }
        
        with open(self.out_model_log, 'w') as target:
            sys.stdout.write("Saving log to {0}".format(self.out_model_log))
            json.dump(details, target, indent=4)
            
    def make_roc_curve(self):
        pass
        
    
if __name__ == "__main__":
    
    #used 3 folds b/c of the limited number of classes
    event_planner = BuildEventPlanner(settings["data"], n_folds=3)
    event_planner.build()
    #raw_input("\nPress enter to quit.")
        
		