# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 14:29:29 2015

@author: awoiz_000

Exploring spatial data
"""
import os
import sys
import pysal
import numpy as np
import pandas as pd
sys.path.append(os.path.join("..", 'crimebusters'))
from conf import settings
from bokeh import plotting
from bokeh.models import HoverTool


plotting.output_file('crime_and_poverty.html')

df = pd.read_csv(settings.crime_data)

TOOLS = "save, hover"

poverty_percentage = u'Per Below Poverty Line'

distance = u'Distance from metro KM'

pov_frame = df[poverty_percentage].replace([np.inf, -np.inf, 0], np.nan).dropna() *100

hist1, edges1 = np.histogram(pov_frame, bins=20)

source1 = plotting.ColumnDataSource(data=dict(count=hist1))

fig1 = plotting.figure(title="Poverty Percentage and Crime Count",
                       tools=TOOLS, background_fill="#E8DDCB")

fig1.quad(top=hist1, bottom=0, left=edges1[:-1], right=edges1[1:],
     fill_color="#036564", line_color="#033649", source=source1)    

hover1 = fig1.select(dict(type=HoverTool))                  
hover1.tooltips=[("count", "@count"),]

fig1.xaxis.axis_label = 'Percentage Poverty Status'
fig1.yaxis.axis_label = 'Crime Count'

hist2, edges2 = np.histogram(df[distance], bins=20)

source2 = plotting.ColumnDataSource(data=dict(count=hist2))

fig2 = plotting.figure(title="Distance From Metro and Crime Count",
                       tools=TOOLS, background_fill="#E8DDCB")

fig2.quad(top=hist2, bottom=0, left=edges2[:-1], right=edges2[1:],
     fill_color="#036564", line_color="#033649", source=source2)  

hover2 = fig2.select(dict(type=HoverTool))                  
hover2.tooltips=[("count", "@count"),]                    

fig2.xaxis.axis_label = 'Distance From Metro (KM)'
fig2.yaxis.axis_label = 'Crime Count'






plotting.show(plotting.vplot(fig1, fig2))

