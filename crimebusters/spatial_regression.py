# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 10:02:35 2015

@author: awoiz_000

This class will load a shapefile, create
a spatial weights matrix, and run a 
geographically weighted regression. 

Data should be visualized as a scatterplot 
matrix first.
"""
import os
import pysal
import numpy as np


class GeographicallyWeightedRegression(object):
    """
    object that loads and analyzes spatial
    data.
    """    
    def __init__(self, in_shapefile, dependent_variable, independent_variables, **kwargs):
        
        self.infile = in_shapefile

        self.dependent_var = dependent_variable
        
        #Checking if input is the correct object. This verification
        #can be applied to all variables        
        if isinstance(independent_variables, list):
            self.independent_vars = independent_variables
        else:
            raise TypeError(message="independent variables must be a list of column headers.")
        
        #uses key word arg            
        self.outfile_summary = kwargs.pop("out_summary", None)
        
        #we can include options. For now
        #just setting one representation
        self.spat_weights = kwargs.pop("spatial_relationship", "queen")
        
        
    def _get_shapefile_dbf(self):
        """
        returns the dbf file path associated with the 
        input shapefile
        """
        
        #name of the file without dir
        file_path = os.path.basename(self.infile).split('.')
        
        #name of the dbf file without dir and add dbf extension
        name = "{0}.dbf".format(file_path[0])
        
        #return full filepath
        return os.path.join(os.path.dirname(self.infile), name)
        
    def _get_dependent_var_array(self,dbf):
        """
        returns the independent variable colmn as a numpy 
        array. Pysal regression requires this
        """
        #grab the single column from the dbf
        dependent = dbf.by_col(self.dependent_var)
        
        #turn it into a numpy type
        dependent_array = np.array(dependent)
        
        #requires nx1 array. Not sure if we need this        
        dependent_array.shape = (len(dependent),1)
        
        return dependent_array
        
    def _get_independent_matrix(self, dbf):
        """
        turns the independent columns into a 
        numpy array for analysis.
        """
        #create a list of columns
        new_array = [dbf.by_col(col) for col in self.independent_vars]
        
        #turn the list of columns into an array, transpose, and return
        return np.array(new_array).T
        
    def _save_summary(self, ols):
        """
        writes the output of the ols to file
        """
        with open(self.save_summary, "w") as target:
            target.write(ols.summary)   
            
    def _get_spatial_weights(self):
        """
        creates the spatial weights object for use in
        OLS. This structure defaults to a certain
        spatial relationship. We can add more key words
        and create different relationships
        """
        #this matrix tells the algorithm how to 
        #look at neighboring features
        #queen looks at polygons with shared edges
        #queen b/c the way the chess piece moves
        if self.spat_weights == "queen":
            return pysal.queen_from_shapefile(self.infile)
            
        else:
            #won't use spatial weights
            return None
            
    def run(self):
        """
        This will run the spatial analysis
        """
        
        #open the shapefile to start working
        #working_file = pysal.open(self.infile)

        #create the spatial weights matrix
        weights = self._get_spatial_weights()
        
        #all shapefiles come with a dbf file
        #this will open that
        dbf_file = self._get_shapefile_dbf()
        
        #open the dbf file for analysis
        open_dbf = pysal.open(dbf_file)
        
        #create the dependent array for OLS input
        dependent_array = self._get_dependent_var_array(open_dbf)
        
        #create the indepent array for OLS input
        indepent_array = self._get_independent_matrix(open_dbf)
        
        #run the OLS
        #This is set up to run Moran's I on the residuals to ensure
        #they are not spatially correlated and White's test. --> need to
        #find out more what that test does.
        ols = pysal.spreg.OLS(dependent_array, indepent_array,
                              w=weights, name_y=self.dependent_var, 
                              name_x=self.independent_vars, name_ds=os.path.basename(self.infile),
                                spat_diag=True, moran=True, white_test=True)
                                
        if self.outfile_summary:
            self._save_summary(ols)
        print ols.summary
        
        open_dbf.close()
        
        
if __name__ == "__main__":
    shapefile=os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","Data",'MetHoodsWithMedCensusCrimeMetDist_Clipped.shp'))
    dependent = "COUNT"
    independent = ["MEANDISTFR","MEANHomeIn", "MEANHomVal","MEANTrvlGr"]
    gwr =GeographicallyWeightedRegression(shapefile,dependent, independent,spatial_relationship="queen")
    gwr.run()