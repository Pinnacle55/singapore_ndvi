# Singapore NDVI Project

This project investigates the use of cloud masks to remove clouds while calculating NDVI over time. This project uses Singapore as a case study.

## Data Collection

Multiband raster data can be collected from a variety of sources, the two most common of which are Landsat 8 and Sentinel 2 data. While Sentinel 2 data has a higher resolution (10 m), A significant amount of Sentinel data is held in long term storage (LTA) and is not always immediately available. Landsat 8 data is more easily acquired and can thus be more suitable despite its lower resolution (30 m).

Both types of data sets can be collected using browser online: Landsat 8 data can be collected from the [USGS earth explorer website](https://earthexplorer.usgs.gov/), while Sentinel 2 data can be collected from the [Copernicus Open Access Hub](https://scihub.copernicus.eu/dhus/#/home). However, both data sets can also be downloaded straight from the command line using their respective APIs.

### Sentinelsat

The API used for Sentinel 2 data is Sentinelsat. The documentation can be found [here]( https://sentinelsat.readthedocs.io/en/stable/). I personally like using Sentinelsat from the command line, but you can also run Sentinelsat from within a Python IDE.

Before running Sentinelsat, ensure that you have provided your Copernicus Open Access hub username and ID as environment variables. Alternatively, you can include them in the command line call itself (see documentation for the appropriate flags).

The command line syntax I used to collect my data is as follows:

`sentinelsat -g singapore_boundary.geojson --sentinel 2 --cloud 5 -d`

Sentinelsat uses a user-defined shape file in order to search for scenes of interest, indicated by the `-g` flag. In general, most shape file formats are allowed, but it is important to make sure that the shape file CRS is in EPSG:4326, as this is the CRS in which Sentinel 2 data is stored. In the command above, I have limited results to Sentinel 2 data as well as data in which there is less than 5% cloud cover.There are also other flags that you can include such as specific periods in which to search (for example, 1st January 2015 to 1st January 2016; see documentation). The `-d` flag indicates that you want Sentinelsat to download your data - this flag can be omitted, in which case Sentinelsat will generate a list of scene IDs that meet the requirements you have outlined.

Sentinelsat data will be downloaded into the folder from which you are running the command line call. In some cases, the Sentinel data will be unavailable as it will be stored in LTA. You will need to wait until it becomes available which may take anywhere between a few hours and a few days. In cases where this delay is untenable, users may wish to use Landsat data instead.

### Landsatxplore



Note: landsatxplore has some compatibility issues with click - specifically, it needs click<8.0, but other geospatial programs require click 8.1.7. Best solution would be to run landsatxplore in a different environment.

landsatxplore was used to collect landsat data for this area. The following command was used: landsatxplore search --dataset landsat_ot_c2_l2 --location lat long --cloud 20
