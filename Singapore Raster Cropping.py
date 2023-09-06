#!/usr/bin/env python
# coding: utf-8

# In[20]:


# Run this in an environment that has gdal installed
import geopandas as gpd
import numpy as np
import os, sys


# In[25]:


study_area = gpd.read_file("singapore_boundary.geojson")

# Remember to switch study area to correct extent
study_area = study_area.to_crs("EPSG:32648")

xmin, ymin, xmax, ymax = study_area.total_bounds


# In[22]:


current_dir = os.getcwd()

# list all files in 20230316
filenames = os.listdir(os.path.join(current_dir, "20230316_IMG_DATA"))

filenames = [os.path.join(current_dir, "20230316_IMG_DATA", file) for file in filenames]

filenames


# In[23]:


# for file in filenames:
#     print(os.path.join(current_dir, "20230316_IMG_DATA", file))


# In[ ]:


# Note that any existing stacked_cropped tif should be deleted before running

# Very interesting, it seems that gdal command line arguments don't work if you provide them with
# os.joined path names that have folders with spaces in them

# You can work around this by manually adding open quotes to the command line call itself
for file in filenames:
    # -tr is a call that sets the resolution of the image
    command4 = 'gdalwarp -t_srs {crs} -te {x_min} {y_min} {x_max} {y_max} -tr 10 10 -r bilinear "{src_file}" "{dst_file}" -co COMPRESS=DEFLATE'
    os.system(command4.format(
        crs = f"EPSG:32648", 
        x_min = np.floor(xmin),
        y_min = np.floor(ymin),
        x_max = np.ceil(xmax),
        y_max = np.ceil(ymax),
        src_file = f"{file}",
        dst_file = f"{file.split('.')[0]}_UPSAMPLED_CROPPED.TIF"
    ))

