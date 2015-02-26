# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 14:29:29 2015

@author: awoiz_000

Exploring spatial data
"""

import pandas as pd
from pandas.tools.plotting import scatter_matrix
import pysal
import numpy as np


def get_data_as_frame(filepath, columns):
    open_dbf = pysal.open(filepath)
    X = []
    for col in columns:
        X.append(open_dbf.by_col(col))
    array = np.array(X)
    open_dbf.close()
    return pd.DataFrame(data=array.T, columns=columns)

print"Running..."
in_data=r'MetHoodsWithMedCensusCrimeMetDist_Clipped.dbf'
cols = ["MEANDISTFR","MEANHomeIn", "MEANHomVal","MEANTrvlGr", "COUNT"]
frame = get_data_as_frame(in_data, cols)
print"making plot..."
scatter_matrix(frame, figsize=(5,5), diagonal="kde")