#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Run this in an environment that has gdal installed
import geopandas as gpd
import numpy as np
import os, sys
from glob import glob

study_area = gpd.read_file("singapore_boundary.geojson")

# Remember to switch study area to correct extent
study_area = study_area.to_crs("EPSG:32648")

xmin, ymin, xmax, ymax = study_area.total_bounds


# In[ ]:


# provide target directory as first argument in command line call
target_dir = sys.argv[1]

band_files = glob(
    os.path.join(
        target_dir,
        "LC08_L2SP*S*_B*.tif",
    )
)

qa_files = glob(
    os.path.join(
        target_dir,
        "*QA_PIXEL.tif",
    )
)

# find all band files and qa files
files = band_files + qa_files


# In[12]:


# Note that any existing stacked_cropped tif should be deleted before running

# Very interesting, it seems that gdal command line arguments don't work if you provide them with
# os.joined path names that have folders with spaces in them

# You can work around this by manually adding open quotes to the command line call itself
for file in files:
    # -tr is a call that sets the resolution of the image
    command4 = 'gdalwarp -t_srs {crs} -te {x_min} {y_min} {x_max} {y_max} -tr 30 30 -r bilinear "{src_file}" "{dst_file}" -co COMPRESS=DEFLATE'
    os.system(command4.format(
        crs = f"EPSG:32648", 
        x_min = np.floor(xmin),
        y_min = np.floor(ymin),
        x_max = np.ceil(xmax),
        y_max = np.ceil(ymax),
        src_file = f"{file}",
        dst_file = f"{file.split('.')[0]}_CROPPED.TIF"
    ))


# In[ ]:




