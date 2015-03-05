# -*- coding: utf-8 -*-
"""
Created on Tue., Feb. 24, 2015

@author: kathleen_llontop
"""
########################################################################
import csv
import time
from sklearn import cross_validation, svm

########################################################################
class BuildEventPlanner(object):
    	
    def __init__(self, filename, classification_field=None):
        self.filename = filename
        self.features = None
        self.classification_field = classification_field or "OFFENSE"
        self.train_time = None
        self.feature_time = None
    		
    def featureset(self):
        """
        load data into the event planner. There are no inputs becase they are 
        grabbed from the object properties.
        
        These are the fields I am working with:
        ['SHIFT', 'OFFENSE', 'METHOD', 'DISTFROMME', 'HomeIncMe', 'TotOccUnit', 'TotalUnits', 'TotVacUnit', 'HomValMed', 'TotComutrs', 'TotPubTran', 'TotTrlTime', 'TrvlGrtr30', 'X', 'Y']
        
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
                
        	
    def train(self, featureset=None):
        """
        This is where the algorithmn will be trained using
        an input featureset. The data should come from
        self.featureset() but the input parameter 
        allows for reuse.
        
        Pass in featureset during cross validation 
        """
        try:
            start = time.time()
            
            featureset = featureset or self.featureset()
            
            #X's the support vectors. These all have to be 
            #numeric values so additional data transformation may be required. 
            #Example error from the initial test ValueError: could not convert string to float: MIDNIGHT
            X = []
            
            #y is the result/classificaiton. Can be string or integer
            y = []
            
            for item in featureset:
                offense, fields = item
                y.append(offense)
                X.append(fields)
            
            #Uses all defaults including an radial basis function (rbf)
            #kernal. Refer to this site: http://scikit-learn.org/stable/modules/svm.html
            
            classifier = svm.SVC()
            
            classifier.fit(X, y)
            
            self.train_time = time.time() - start 
            
            return classifier
        
        except ValueError as error:
            #SVM will throw this if all the support vectors are not numeric
            raise ValueError(error)
        
    
    def build(self):
        """
        This is where the class will build the output and write
        to disk likely using pickle. """
        pass

    def cross_validate(self):
        """
        A function to test the accuracy of the 
        model.
        """
        #clf = svm.SVC(kernel='linear', C=1)
        #scores = cross_validation.cross_val_score(clf, X_train, y_train, cv=k)
    
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
    infile = r'C:/Users/andrew_woizesko/Documents/Georgetown Data Science/Capstone/Data/Crime_for_classifier.csv'
    this_build = BuildEventPlanner(infile)
    cls = this_build.train()
    #raw_input("\nPress enter to quit.")
        
		