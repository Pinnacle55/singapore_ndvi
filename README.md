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

## Data Preprocessing

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
    # directory. it returns a list of file paths which you can then use as the input for es.stack 
    # in order to stack the bands into a multi band raster.
    
    # note that this expects a list of polygons, so you need to put it in a list even if its just the
    # one polygon
    band_paths = es.crop_all(
        paths_to_crop, output_dir, [target_shapefile], overwrite=True
    )
    
    band_paths_list.append(band_paths)
```

The `es.crop_all` function returns a list of all the file paths to the cropped rasters. This list can be easily fed into the `es.stack` function, which stacks all of the files in the given file paths into a single multiband raster. The `out_path` argument tells the function where you want to save this multiband raster.
