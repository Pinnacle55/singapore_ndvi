# Singapore NDVI Project

This project investigates the use of cloud masks to remove clouds while calculating NDVI over time in each administrative zone. This project uses Singapore as a case study.

![alt text](https://github.com/Pinnacle55/singapore_ndvi/blob/21cdd54fbf4aa55f0ac759e48e97c2c7ac91b59a/Images/Singapore%20S2%20Cloudless%20Cloud%20Mask.png?raw=True "Cover Page")

## Data Collection

Multiband raster data can be collected from a variety of sources, the two most common of which are Landsat 8 and Sentinel 2 data. While Sentinel 2 data has a higher resolution (10 m), A significant amount of Sentinel data is held in long term storage (LTA) and is not always immediately available. Landsat 8 data is more easily acquired and can thus be more suitable despite its lower resolution (30 m).

Both types of data sets can be collected using browser online: Landsat 8 data can be collected from the [USGS earth explorer website](https://earthexplorer.usgs.gov/), while Sentinel 2 data can be collected from the [Copernicus Open Access Hub](https://scihub.copernicus.eu/dhus/#/home). However, both data sets can also be downloaded straight from the command line using their respective APIs.

### Sentinelsat

The API used for Sentinel 2 data is Sentinelsat. The documentation can be found [here]( https://sentinelsat.readthedocs.io/en/stable/). I personally like using Sentinelsat from the command line, but you can also run Sentinelsat from within a Python IDE.

Before running Sentinelsat, ensure that you have provided your Copernicus Open Access hub username and ID as environment variables. Alternatively, you can include them in the command line call itself (see documentation for the appropriate flags).

The command line syntax I used to collect my data is as follows:

```sentinelsat -g singapore_boundary.geojson --sentinel 2 --cloud 5 -d```

Sentinelsat uses a user-defined shape file in order to search for scenes of interest, indicated by the `-g` flag. In general, most shape file formats are allowed, but it is important to make sure that the shape file CRS is in EPSG:4326, as this is the CRS in which Sentinel 2 data is stored. In the command above, I have limited results to Sentinel 2 data as well as data in which there is less than 5% cloud cover.There are also other flags that you can include such as specific periods in which to search (for example, 1st January 2015 to 1st January 2016; see documentation). The `-d` flag indicates that you want Sentinelsat to download your data - this flag can be omitted, in which case Sentinelsat will generate a list of scene IDs that meet the requirements you have outlined.

Sentinelsat data will be downloaded into the folder from which you are running the command line call. In some cases, the Sentinel data will be unavailable as it will be stored in LTA. You will need to wait until it becomes available which may take anywhere between a few hours and a few days. In cases where this delay is untenable, users may wish to use Landsat data instead.

### Landsatxplore

The first and most pressing issue is that landsatxplore has some compatibility issues with `click`. Specifically, it needs `click<8.0`, but other geospatial programs require `click 8.1.7`. The best solution would be to run landsatxplore in its own specific environment.

Once again, landsatxplore can be used from the command line. After providing your USGS username and password credentials as environment variables, I used the following syntax in the command line:

```landsatxplore search --dataset landsat_ot_c2_l2 --location 1.3521 103.8198 --cloud 20 > results.csv```

The command line call above searches for all datasets that meet the above conditions. The `--dataset` flag indicates what kind of data you would like to download (for example, level 1 or level 2 products; See documentation for examples), the `--location` flag is a point that you want to be contained in the scene provided in latitude and longitude, and the `--cloud` flag indicates the maximum allowed cloud cover. `> results.csv` indicates that you wish to save the results of this query in a CSV file.

Downloading landsat scenes using landsatxplore is actually slightly more complicated than you would expect. The most pressing issue is that there appears to be an issue with landsatxplore error handling: in some cases, it will claim that the download failed when it actually succeeded. In these cases, however, the program will still throw an error and cease to continue with the rest of the scenes. You should use a try except block as a workaround as shown in the following code block:

```
from landsatxplore.earthexplorer import EarthExplorer

# Sign in to earthexplorer
ee = EarthExplorer("username", "password")

# Read the aforementioned csv and get the scene IDs
with open("landsat_IDs.csv") as f:
    lines = [line.rstrip('\n') for line in f]
f.close()

for scene in lines:
    # note that there appears to be an issue where the program will claim that the download failed but 
    # it actually succeeded - despite this the program will throw an error and the script will not
    # continue. As a workaround use a try except block to continue the loop despite the error being 
    # thrown.

    try:
        ee.download(scene, output_dir='./data')
    except:
        print(f"{scene} failed to download!")
        continue

ee.logout()
```

Nevertheless, Landsat data is almost always available, allowing you to start working on it almost immediately.

## Singapore Suburbs Shapefile - Data Collection and Cleaning

[OpenStreetMaps (OSM)](https://www.openstreetmap.org/) is probably the best place to go to collect data on administrative areas, transport routes, etc. Although they are more likely to be higher resolution shape files that can be found on governmental websites in specific countries, etc. OSM has an extremely wide reach. 

The OSM API can be a bit tricky to use - thankfully, there is a service called [Overpass Turbo](https://overpass-turbo.eu/) that makes querying and downloading OSM data much easier. My general workflow is to use the OSM website to find the specific feature that I'm interested in (for example, a specific administrative boundary level or a specific rail line) and then using Overpass Turbo to download that data over the entirety of the study area.

However, because the OSM data is user-submitted, it is important to clean it beforehand in order to make sure that it is usable for our future analyses. The shapefile cleaning workflow can be found in `Singapore Geopandas Cleaning.ipynb`.
The first thing to do is to switch the CRS of the zip file to an appropriate CRS. Ideally, you want the CRS to be in a reference system measured in meters. 


```
# note that this is in epsg 4326
sg = gpd.read_file("singapore.geojson")

# First, convert to UTM 48N to get values in meters
sg = sg.to_crs("EPSG:32648")
```

You can then start the cleaning process - for example OSM data often comes with a lot of information that is not particularly useful, such as the name of the Polygon in a different language. In most cases you can subset the columns of the Geopandas dataframe into just `name` and `geometry`. OSM data may also contain duplicates, which should be dropped.
In general, it is good to constantly visualize your shapefile to identify any potential errors. For example, a quick visualization of the OSM data reveals that there is a suburb missing. It turns out that the area in question had been subsumed into an adjacent suburb a few years prior.

```
sg_suburbs = sg[sg["place"] == "suburb"][["name", "geometry"]]

# quick visualization
fig, ax = plt.subplots(figsize = (20,24))
sg_suburbs.plot(ax = ax, edgecolor = "black")

# quick and easy way to annotate names of administrative areas
sg_suburbs.apply(lambda x: ax.annotate(text = x["name"], 
                                       xy = x["geometry"].centroid.coords[0],
                                       ha = "center"), axis = 1)
plt.show();
```

![alt text](https://github.com/Pinnacle55/singapore_ndvi/blob/e2a85064a101a4fae224aa6925a371ae6b659664/Images/Singapore%20Shapefile%20Cleaning%20-%20Missing%20Area.png?raw=True "Missing data")

Another quick and easy way to find missing data is to run `sg_suburbs.unary_union`. This generates a Polygon that is very good at highlighting missing areas as shown below:


![alt text](https://github.com/Pinnacle55/singapore_ndvi/blob/e2a85064a101a4fae224aa6925a371ae6b659664/Images/Singapore%20Unary%20Union%20-%20Missing%20Data.png?raw=True "Unary union")

We can see that not only is there a small area missing but there are very small gaps between polygons, likely between the edges of adjacent polygons - these should be filled in prior to analysis.

The first thing we should do is fix the missing suburb. To do this we manually create a rectangular Polygon that covers the entirety of the missing suburb. The coordinates of this Polygon can be easily found from the above visualization. We can then find the difference between this rectangular Polygon and the unary union of the rest of the suburbs to get a Polygon of the missing suburb.

```
from shapely.geometry import Polygon
# We manually create a rectangular polygon
# Define the coordinates of the polygon's exterior ring
# the coords are given by bottom right, top right, top left, bottom left
coordinates = [(362000, 147000), (362000, 149500), (364500, 149500), (364500, 147000)]
manual_poly = Polygon(coordinates)
# We then find the difference between this polygon and the unary_union of the entire dataset
ty_poly = manual_poly.difference(sg_suburbs.unary_union)
```

We can then combine `ty_poly` (the polygon of the missing suburb) with the Polygon of the suburb that we want to merge it into. It is then simply a case of replacing the original geometry with the merged Polygon.

```
# Combine the old BT poly and the TY poly into a new merged polygon
from shapely.ops import unary_union
merged_poly = unary_union([ty_poly, bt_poly.iloc[0]])
# Set the merged poly as the new BT poly
sg_suburbs.loc[sg_suburbs["name"] == "Bukit Timah", "geometry"] = merged_poly
```

The next step is to get rid of the little gaps between polygons. These gaps are most likely because the polygons were not traced onto each other; indeed, you can't even see these gaps on the visualization shown above.

The way I fixed this is to first create a Polygon of the boundary of the entire dataset called `sg_poly`. The difference between this and the unary union of the rest of the data gives us a multiPolygon of all the `missing` areas. For each Polygon in the missing multiPolygon, we find the suburb that is closest to that missing Polygon by calculating the distance from that Polygon to the centroid of every suburb in Singapore. We then combine the missing Polygon with the geometry of the closest suburb. We then replace the geometry of that suburb with the merged Polygon.

```
# Full polygon of singapore based on sg_suburbs
sg_poly = Polygon([i for i in sg_suburbs.unary_union.exterior.coords])
# multipolygon of all the missing bits
missing = sg_poly.difference(sg_suburbs.unary_union)
# for each missing polygon, find the nearest suburb (using centroid), then merge the two polygons
for polygon in missing.geoms:
    # This finds the geometry of the suburb that is CLOSEST to the missing polygon,
    # and merges them
    merged = unary_union([sg_suburbs.loc[sg_suburbs.centroid.distance(polygon).idxmin(), "geometry"], polygon])
    
    # Replace the original suburb polygon with the merge
    sg_suburbs.loc[sg_suburbs.centroid.distance(polygon).idxmin(), "geometry"] = merged
```

An issue with this method is that it can sometimes leave extremely small missing polygons (on the order of centimeters) due to floating point errors with the unary union function. Thankfully this can be rectified very easily by buffering all the polygons by an extremely small amount (1 meter) and then unbuffering it by that same amount.

```
# buffer then unbuffer, save to original polygon
sg_suburbs.loc[sg_suburbs.centroid.distance(missing).idxmin(), "geometry"] = sg_suburbs.loc[sg_suburbs.centroid.distance(missing).idxmin(), "geometry"].buffer(1).buffer(-1)
```

Using this cleaned Singapore suburb shape file I generated 2 GeoJSON files: ` singapore_boundary.geojson`, Polygon of the boundary of the study area, and `sg_suburbs_cleaned.geojson`, which contains the geometry and the name of all the suburbs in Singapore. The former will be used for cropping our Sentinel and Landsat images, while the latter will be used for zonal statistics.

## Data Preprocessing - Raster Cropping

Regardless of whether you are using Sentinel 2 or Landsat 8 data, you should always conduct some basic preprocessing in order to reduce the amount of storage required for your projects as well as ensure that your projects are in a file format that is more conducive to subsequent analysis.

The first thing that you should do is crop your data to your study area - this can significantly reduce the size of your raster files and makes future analyses much more computationally efficient.

You can do this one of two ways: the first thing you can do is use gdal command line functions. Gdal is extremely powerful - honestly, you can pretty much do anything you want to a raster file using gdal. However, because it is a primarily command line based program (that there are Python wrappings for gdal but they are quite complicated), you will have to run the scripts from the command line using Anaconda prompt.

The second way you can do it is to use `EarthPy`. This is a module that makes cropping and stacking rasters extremely easy. It does not have the fine control that gdal has — for example, you can't control the output resolution nor the target CRS — but in cases where your data comes from the same source such as the Earth Explorer database or the Copernicus SciHub, `EarthPy` takes only one or two lines of code to crop and stack your rasters.

The first step to cropping is to identify the area that you want to crop your raster to - in the case of `EarthPy` and `rasterio`, you will need to provide a shapefile that describes the extent of the area that you want to crop to. If you are using gdal, then you can provide this extent either in coordinates or as a shapefile.

In this case we are going to use the `singapore_boundary.geojson` shapefile we generated in the previous section. After loading it in using Geo pandas the first thing we need to do is to make sure that the CRS of this shape file is the same as the CRS of the raster files. In this case, I already know the CRS that I want to target, but in cases where you don't know the CRS, you will need to load in your raster data, identify the CRS in the profile metadata, then reproject your shapefile to that CRS.

Following that, you can use the shapefile as is. In cases where the shape file is not rectangular, `EarthPy` will crop the raster to a rectangular area and anything outside the shape file will be assigned a nodata attribute. In my case, however, I want to crop the rest to a rectangular area. I can get the corner coordinates of my shape file by using the `total_bounds` attribute. I can then manually use shapely's box function to create a rectangular Polygon.

```
from shapely.geometry import box
# Set target_shapefile to the appropriate coordinates
# Create a box from the total_bounds of the shapefile
target_shapefile = gpd.read_file("singapore_boundary.geojson")
target_shapefile = target_shapefile.to_crs("EPSG:32648")
xmin, ymin, xmax, ymax = target_shapefile.total_bounds
target_shapefile = box(np.floor(xmin), np.floor(ymin), np.ceil(xmax), np.ceil(ymax))
```

The following example uses landsat data, but this workflow can be easily applied to Sentinel data. Once we have our shape file, we need to get the file paths of all the rasters that we want to crop. We can do this using `glob` and `os.path`. Assuming that you have identified the filepaths of all rasters that you want to crop (remember, these rasters must have the same size, CRS, and resolution), you can run the simple `es.crop_all` function to crop all of the rasters.

```
import earthpy.spatial as es
# I want to crop both band data as well as the QA pixel raster for cloud detection
paths_to_crop = stack_band_paths + stack_qa_paths

# This function will crop all of the specified bands and write them into the specified output 
# directory. it returns a list of file paths which you can then use as the input for es.stack()
# in order to stack the bands into a multiband raster.

# note that this expects a list of polygons, so you need to put it in a list even if its just the
# one polygon
band_paths = es.crop_all(
    paths_to_crop, output_dir, [target_shapefile], overwrite=True
)

band_paths_list.append(band_paths)
```

The `es.crop_all` function returns a list of all the file paths to the cropped rasters. This list can be easily fed into the `es.stack` function, which stacks all of the files in the given file paths into a single multiband raster. The `out_path` argument tells the function where you want to save this multiband raster.

```
for scene in band_paths_list:
    # Note that we use folder[:-1] - we don't want to add the QA PIXEL raster to our stacked raster
    stack, metadata = es.stack(scene[:-1], out_path = out_path)
```

We have now successfully cropped the extremely large LANDSAT scenes to our  study area. As mentioned, this significantly decreases the file size on disk (in my case, it reduced it from 800 MB for the entire dataset to approximately 30 MB for the stacked multiband raster).

## Data Preprocessing - Cloud Detection

One of the major issues with satellite imagery is the presence of clouds. Clouds can obscure the surface from view, preventing the analysis of the ground itself. This is often why cloud cover is included in the metadata of both Sentinel and LANDSAT images. Consequently, the removal of clouds is an important preprocessing step that allows researchers to get a clearer view of the study area in order to make conclusions, especially in time series analyses. 

At the present day, clouds and cloud shadow (CCS) detection has improved by leaps and bounds due to the wealth of available data as well as improvements in machine learning technology, which allows algorithms to more accurately and precisely predict the presence of clouds in a data set.

Depending on whether you are using Sentinel or LANDSAT data your options for cloud detection will be different. In the case of Sentinel data, there are several third party tools that can be used for cloud detection; in this particular case I chose to use S2cloudless, which is a relatively easy to use module that can be run on any Sentinel scene. It is significantly better at cloud detection than the cloud mask packaged with Sentinel 2 products.

It is much easier to do cloud detection on LANDSAT images since LANDSAT Level 2 products automatically come bundled with a QA PIXEL raster that contains information about the scene, including clouds cloud shadow and water areas. LANDSAT data is processed using the CFMask algorithm, which is considered to be relatively robust (although more comprehensive cloud masking algorithms have since been developed). Consequently, the relatively robust QA pixel raster that comes with LANDSAT data generally precludes the need for the use of third party tools.

### Cloud Detection for Sentinel Data - S2Cloudless

First, let's take a look at the cloud mask that comes packaged with Sentinel data. I'm not entirely sure where this cloud mask comes from or how Sentinel creates this cloud mask, but it's a good practice just to take a look and see how good/bad it is.

The Sentinel cloud mask is a multiband raster containing several layers - each layer is a binary mosque that indicates the presence of clouds, cirrus, or snow. In our case, since we are looking at Singapore a tropical country we can pretty much combine all of these layers into a single layer without losing too much information.

```
# Read in the cloud mask
dataset = rio.open("20230316_IMG_DATA\\MSK_CLASSI_B00_UPSAMPLED_CROPPED.tif")
data = dataset.read()

# We'll mask anything that is present in any layer
# If there is a 1 (i.e., positive detection) in any layer, the mean along the 0th-axis will be > 0.
data_means = np.mean(data, axis = 0)

# Set all pixels with means > 0 to 1. This creates binary masks that indicates the presence of any contaminant (cloud, snow, or cirrus).
data_means[data_means != 0] = 1

show(data_means)
```

![alt text](https://github.com/Pinnacle55/singapore_ndvi/blob/4dc8228f55336beee7ddb3ff3bee1c23a6dffb97/Images/Singapore%20TCC.png?raw=True "Singapore Original")

![alt text](https://github.com/Pinnacle55/singapore_ndvi/blob/4dc8228f55336beee7ddb3ff3bee1c23a6dffb97/Images/Sentinel%20Default%20Cloud%20Mask.png?raw=True "Sentinel Default Cloud Mask")

It's pretty clear that the Sentinel cloud mask is not particularly good. We can visualize this in a more direct way by overlaying the cloud mask on top of the true color composite in order to see which clouds the cloud mask detected.

```
# Compare to the cloud mask provided by sentinel2 itself

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
ax.imshow(true_color_image)
ax.imshow(data_means, cmap="gnuplot", alpha=0.3 * data_means)
```

![alt text](https://github.com/Pinnacle55/singapore_ndvi/blob/4dc8228f55336beee7ddb3ff3bee1c23a6dffb97/Images/Singapore%20Sentinel%20Cloud%20Mask.png?raw=True "Sentinel Default Cloud Mask")

It is clear that we cannot rely on the cloud mask provided by Sentinel itself. Thankfully, there are several easy-to-use cloud detection algorithms that can be used for Sentinel data. One of these is [S2cloudless](https://github.com/sentinel-hub/sentinel2-cloud-detector), which uses deep learning algorithms to identify clouds in Sentinel data. S2cloudless can either be used on data downloaded from site habit itself, or it comes prepackaged as an available layer when you download data from Sentinel Hub. It is important to note, however, that S2 cloudless does not detect the presence of cloud shadow.

```
from s2cloudless import S2PixelCloudDetector, download_bands_and_valid_data_mask

# Instantiate cloud detection model
cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2, all_bands=True)

# Import stacked raster of all Sentinel bands - it is important to note that s2cloudless uses reflectances for its calculations, so the Sentinel bands must be converted to reflectances beforehand.
# In addition, it is EXTREMELY important to note that after October 26, 2021, Sentinel changed their DN-reflectance calculation such that reflectances = (data - 1000) / 10000. Prior to this date, the algorithm was reflectance = data / 10000
src = rio.open("SINGAPORE_20230316_STACKED.tif")
dataset = src.read()

# Move the band axis to the back
dataset_image = rio.plot.reshape_as_image(dataset)

# Get cloud probability maps
cloud_prob = cloud_detector.get_cloud_probability_maps(dataset_image[np.newaxis, ...])
```

![alt text](https://github.com/Pinnacle55/singapore_ndvi/blob/4dc8228f55336beee7ddb3ff3bee1c23a6dffb97/Images/Singapore%20S2%20Cloudless%20Cloud%20Mask.png?raw=True "Sentinel S2Cloudless Mask")

You can see that the S2 cloud mask is much better than the default cloud mask provided by Sentinel. Furthermore, the S2 cloud mask algorithm allows you to specify your own threshold regarding how aggressive or conservative you would like the program to be when detecting clouds. In this particular case I just used the default values and that works quite well.

As previously mentioned, however, s2cloudless does not detect the presence of cloud shadow - depending on your ultimate goals this may or may not be an issue. 

### Cloud Detection for LANDSAT Data - QA Pixel Rasters

Cloud detection for LANDSAT data is much simpler than Sentinel data because the default cloud mask packaged with LANDSAT Level 2 products is relatively good. This information is contained within the QA pixel raster, which is actually a 16-bit raster that encodes a significant amount of information about the scene. Specifically, the 16 bits are used to code the presence of clouds snow cirrus and water as well as the confidence of the algorithm with respect to each of these contaminants. 

To utilize this QA pixel raster, you will need to encode the 16-bit DNs into binary and then use the [lookup table provided by LANDSAT](https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1619_Landsat8-9-Collection2-Level2-Science-Product-Guide-v5.pdf) to interpret the results. While this can seem a bit annoying, this actually makes it relatively easy for you to define your own algorithm - for example, you can create a function that parses the QA pixel raster to create a binary mask that masks out water and clouds only, leaving snow unmasked. Furthermore, the fact that the 16 bit data also encodes confidence levels allows you to generate more aggressive or more conservative cloud masks depending on your needs. 

```
def qa_pixel_interp_conservative(number):
    '''
    Helps interpret the 16bit data in the landsat qa pixels
    
    returns True if there is high confidence cirrus, snow/ice, cloud shadow, OR clouds
    '''
    binary = bin(number)[2:].zfill(16)
    
    # if high confidence cirrus, snow/ice, cloud shadow, and clouds
    # 01 - low, 10 - medium, 11 - high
    if int(binary[:2]) > 10:
        return True
    elif int(binary[2:4]) > 10:
        return True
    elif int(binary[4:6]) > 10:
        return True
    elif int(binary[6:8]) > 10:
        return True
    else:
        return False
```
![alt text](https://github.com/Pinnacle55/singapore_ndvi/blob/4dc8228f55336beee7ddb3ff3bee1c23a6dffb97/Images/Landsat%20TCC.png?raw=True "Landsat TCC")

![alt text](https://github.com/Pinnacle55/singapore_ndvi/blob/4dc8228f55336beee7ddb3ff3bee1c23a6dffb97/Images/Landsat%20Cloud%20Mask%20Conservative.png?raw=True "Conservative Cloud Detection")

You'll notice from the images above that the LANDAST QA pixel band does not only detect clouds but also detects cloud shadows. In fact, you can even use the QA pixel band to remove any water bodies that were detected by FMask.

## Preprocessing - Cloud Imputation / Mosaicking

For some use cases, it is sufficient to have a binary cloud mask raster. You can then create a masked array using your cloud mask in order to remove any pixels that are affected by clouds. However, in some cases this is insufficient for your analysis. For example, in time series analyses, you want to see how the entire scene changes over time - if there are clouds obstructing your view of the scene, then you would not be able to make any good conclusions about how the landscape has changed over time.

In cases where you need to have a full, clear view of the scene you will need figure out some way of imputing the missing data (i.e., the areas covered by clouds). In some cases, you can use an average of the surrounding pixels, or you can use some interpolation methods that are available in modules such as scipy. However, in cases where the missing areas are exceptionally large such as in very cloudy areas like Singapore, these interpolation methods would introduce serious biases. 

To get around this, you can use a technique known as mosaicking - this is a process by which multiple overlapping images are stitched together to form a final image. Mosaicking is not only used in cloud removal processes - there can be any number of reasons why you would want to mosaic an image such as improving the resolution of an image, replacing certain parts of an image that have higher quality data, or filling in nodata gaps. In this case, we will be using our cloud masks to identify areas that need to be patched in an image and replacing those areas with bits of another image in which those areas are not covered by clouds.

First, we need to download the data to be mosaicked. This requires us to get multiple images of the same scene over different timescales. This really depends on your use case as well as the availability of data - for example, if you're looking at yearly data, then your images can be months apart. However, if you're looking at seasonal data, then you want your images to be at most weeks or even days apart. This may not be possible depending on your data source: for example, LANDSAT images are taken once every 17 days.

Once you've generated multiband rasters for each of your images, stack the multiband rasters into a 4D numpy array (the four dimensions are date, spectral band, height, width). Note that the order of the array matters - ideally you want to fill in missing data with data from a scene that was as close to the original scene (temporally) as possible (i.e., you want to fill in the image with bits of an image taken 3 days later rather than 7 days later); I'll refer to this as the `stacked_mb_raster`. Similarly, stack the QA pixel rasters into a three-dimensional numpy array (only 3D because the QA pixel raster is comprised only of a single band) - we refer to this as the `stacked_qa_raster`. 

We then turn the 3D QA pixel raster into a binary 3D cloud mask by running our parser function - anything flagged as a contaminant is given a value of 1, while clean pixels are given a value of 0.

```
# classify the QA raster using conservative cloud estimates
unique_vals = np.unique(stacked_qa_raster)          # find all unique 16bit values
masked_vals = apply_array_func(func, unique_vals)   # runs the parser on each unique element in the array
masked_vals = unique_vals[masked_vals]              # get a 1D array of all 16bit values we'd like to mask
cl_mask = np.isin(stacked_qa_raster, masked_vals)   # create the binary mask based on the flagged values identified in the previous step
```

We now have a stacked cl_mask (date, height, width) containing a binary cloud mask for each multiband raster in our study period. 

Convert the stacked multiband rasters into np.float32 - LANDSAT data comes as int16, which does not accept np.nan as a nodata value; converting it to np.float32 fixes this issue. We will then mask the `stacked_mb_raster` with the `cl_mask` that we generated, giving us a masked 4D numpy array. We then down the array layer by layer (i.e., date by date), and we will impute any missing data with the value of the pixel in the scene below it (assuming that the pixel is also not missing). This sounds complicated but is easily accomplished.

```
# The base scene is the first layer (2022-04-01)
base_scene = masked_mb_raster[0, ...]

# for each scene
for layer_num in range(masked_mb_raster.shape[0] - 1):
    # Takes the base scene and fills the missing with stuff from layer 2
    result = np.ma.filled(base_scene, 
                          # this fills missing data in layer 2 with nans
                          # if there is missing data in the same coord in both layers, then 
                          # then the result array will have a np.nan in that coord
                          np.ma.filled(masked_mb_raster[layer_num + 1, :], fill_value = np.nan)
                         )
    
    # 'result' is an unmasked array containing np.nan values. We want to turn result back into a masked array,
    # where np.nan are the values to be masked.
    result_mask = np.isnan(result)

    # Rename it base_scene for the loop to work
    base_scene = np.ma.masked_array(result, result_mask)
```

Eventually, the `base_scene` will be a single masked 3D raster with as many pixels filled as possible. It is important to note that there may be some pixels are remain unfilled - these are pixels that were not clear in any of the images. The results of the mosaicking can be seen below:

![alt text](https://github.com/Pinnacle55/singapore_ndvi/blob/4dc8228f55336beee7ddb3ff3bee1c23a6dffb97/Images/Landsat%20TCC.png?raw=True "Landsat TCC")

![alt text](https://github.com/Pinnacle55/singapore_ndvi/blob/7d9578ca877d220811b166d175888a17e86f364c/Images/Singapore%20Imputed%20Clouds.png?raw=True "Singapore Imputed Clouds")

As you can see, the mosaicking has significantly improved the quality of the image and helped to remove several clouds. Note that there are still some areas with missing data, but overall you are able to get a much better picture of the scene compared to the unprocessed image.

## NDVI Calculations

The normalised difference vegetation index (NDVI) is classic spectral index used to quantify the health and density of vegetation using multispectral sensor data. It is a popular index because it is easy to calculate – requiring only the red and near infrared bands – and is also very easy to interpret. The NDVI has a value between -1 and 1. Areas with healthy vegetation will have an NDVI of 1, areas without vegetation have an NDVI of 0, and water bodies will have a negative NDVI value.

The NDVI is calculated as follows: (NIR - Red) / (NIR + Red), where NIR and Red refer to the reflectance of the near-infrared and red bands, respectively. This is easily calculated by slicing the stacked multiband raster. Note that if you have not converted the DN numbers to reflectances, you will need to do so at this step. This is a relatively simple function as shown [here](https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1619_Landsat8-9-Collection2-Level2-Science-Product-Guide-v5.pdf):

```
ndvi_dict = {}
years = [str(year) for year in range(2013,2023)]

for idx, image in enumerate(images):
    with rio.open(image) as src:
        # import red and nir bands
        nir = src.read(5, masked = True)
        red = src.read(4, masked = True)
    src.close()

    # Adjust to reflectance using the given scale factors
    nir = nir * 0.0000275 + -0.2
    red = red * 0.0000275 + -0.2

    # ndvi calculation
    ndvi = (nir - red) / (nir + red)

    # Landsat 8 Level 2 products have an issue where water can result in negative reflectance
    # in such cases, set NDVI to -1
    ndvi[(nir < 0) | (red < 0)] = -1
    
    # Append raster to dictionary
    ndvi_dict[years[idx]] = ndvi
```

One important thing to note is that LANDSAT Level two products have an unusual property where water can sometimes result in negative reflectances. Well this is not an issue for NDVI analyses, it is important to keep in mind for other analysis–for example, if you are looking to determine sediment content in water using satellite imagery. In such cases, you may want to use Level 1 data and manually transform it using top-of-atmosphere reflectance calculations.

NDVI maps generally look like the following:

![alt text](https://github.com/Pinnacle55/singapore_ndvi/blob/e3cbee49ea4a2aac9cac6a3bd40803b8c08c175a/Images/Singapore%20NDVI%20Landsat%20Masked.png?raw=True "NDVI")

In this particular case, I masked out both the clouds and any water bodies. You can see that green areas indicate areas where there is lush vegetation and red areas indicate locations where no vegetation is growing. Note that the scale on which you present your NDVI data makes a big difference - in this particular case I used a scale from 0 to 1 even though and DVI can range from -1 to 1. Without this modified scale, area without vegetation would look yellow, while water bodies will be red. This particular scale thus more clearly highlights areas without vegetation.

## Presenting Data - Animations

One of the most compelling ways of presenting time series data is through the use of animations. This is much more effective than a series of graphs at showing the changes in the scene over time. Using matplotlib to animate graphs in Python can be a little frustrating, but it is relatively simple using matplotlib's inbuilt animation classes.


