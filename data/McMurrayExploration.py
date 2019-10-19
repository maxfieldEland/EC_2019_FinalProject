#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 08:45:47 2019
Fort McMurray Data Exploration
@author: max
"""

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import datetime 
import os

fortMShape = gpd.GeoDataFrame.from_file('nyu_2451_35506/contourl.shp')
# restrict the latitude and longitude to the fire area
# lat range = 56-57
#long range = -112 - -110

fortM = gpd.read_file('DL_FIRE_V1_79808/fire_archive_V1_79808.shp')

#tile id == 33 aligns the lattitude, but need lower longitude

fortM.plot(ax = base,marker = 'o',column = 'BRIGHT_TI4', figsize = (20,20),cmap = "Reds" )
plt.show()
# Build list of fire dates
dates = pd.unique(fortM.ACQ_DATE)
# transpose df to plot by date column

# add fire locations dynamically as time passes
for date in dates:
    #base = fortMShape[(fortMShape['tile_id'] == 27 )| (fortMShape['tile_id'] == 33) ].plot(color = 'black', figsize = (20,20))
    date_df = fortM[fortM['ACQ_DATE'] == date]
    fig = date_df.plot(ax = base,column = 'BRIGHT_TI4',cmap = "Reds", figsize = (10,10))
    plt.show()
    fig.axis('off')
    
    
    fig.annotate(date,xy=(0.1,0.225),xycoords = 'figure fraction',
            horizontalalignment='left', verticalalignment='top',
            fontsize=35)
    filepath = 'DL_FIRE_V1_79808/gif_jpgs/'+ date+'_fire_gif.jpg'
    chart = fig.get_figure()
    chart.savefig(filepath, dpi=300)

   
