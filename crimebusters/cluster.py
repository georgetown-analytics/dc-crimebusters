"""
Creates a colorized kmeans cluster of crime in Washington, DC.
Attempted to add pop-ups but there are too many co-located
points. Data for kmeans is all numeric except for the input
crime labels. All data was normalized and coordinates are 
in WGS 84. Data will not load into ArcGIS online - the site
usually crashes. 
"""

import os
import time
import numpy as np
import pandas as pd
from conf import settings
from datetime import datetime
from bokeh.charts import Scatter
from sklearn.cluster import KMeans
from collections import OrderedDict
from bokeh.plotting import output_file
from sklearn.preprocessing import Imputer

#Create a timestamp for the output file
def time_stamp():
    """
    Creates mm-dd-yyyy timestamp
    """
    now = time.time()
    return datetime.fromtimestamp(now).strftime("%m%d%Y")
    
def create_ordered_dict(data_frame, group_by, x_field="X", y_field="Y"):
    """
    puts data into a dicitonary for Bokeh. 
    """
    pdict = OrderedDict()
    
    g = data_frame.groupby(group_by)
    
    for i in g.groups.keys():
        lat = getattr(g.get_group(i), y_field)
        lng = getattr(g.get_group(i), x_field)
        pdict[i] = zip(lat,lng)
    return pdict
    
def main():
    """
    This function will
        -load data from a csv
        -impute missing data with the column's mean
        -perform kmneans clustering
        -produce an html scatter plot
    """
    
    #load data from a CSV to a dataframe
    crime_data = pd.DataFrame.from_csv(settings["crime_data"], sep=',')
    
    #get all column headers
    headers = crime_data.columns
    
    #load all numeric data into an array. The offense column from the crime data
    #is excluded
    as_array = np.asfarray(crime_data[headers[1:]])
    
    #number of groups
    n_clusters=40
    
    #Correct missing data 
    imputer = Imputer(missing_values=-999, strategy="mean")
    patched = imputer.fit_transform(as_array)
    
    #cluster data 
    cluster = KMeans(n_clusters=n_clusters)
    cluster.fit(patched)
    
    #assigned grouped labels to the crime data
    labels = cluster.labels_
    crime_data["labels"]=labels
    
    pdict = create_ordered_dict(crime_data, "labels")
    
    #location of output graph
    file_name = os.path.join("..", 'tests', "kmeans_clusters_{0}.html".format(time_stamp()))
    output_file(file_name)
    
    #create out graph
    TOOLS="pan,wheel_zoom,box_zoom,reset"
    scatter = Scatter(pdict.values(), title="Crime Clusters", filename=file_name, tools=TOOLS)
    scatter.show()


if __name__ == "__main__":
    main()