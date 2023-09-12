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

## Data Preprocessing

Regardless of whether you are using Sentinel 2 or Landsat 8 data, you should always conduct some basic preprocessing in order to reduce the amount of storage required for your projects as well as ensure that your projects are in a file format that is more conducive to subsequent analysis.

The first thing that you should do is crop your data to your study area - this can significantly reduce the size of your raster files.
