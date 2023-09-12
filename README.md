# Singapore NDVI Project

This project investigates the use of cloud masks to remove clouds while calculating NDVI over time. This project uses Singapore as a case study.

## Data Collection

Multiband raster data can be collected from a variety of sources, the two most common of which are Landsat 8 and Sentinel 2 data. While Sentinel 2 data has a higher resolution (10 m), A significant amount of Sentinel data is held in long term storage (LTA) and is not always immediately available. Landsat 8 data is more easily acquired and can thus be more suitable despite its lower resolution (30 m).

Both types of data sets can be collected using browser online: Landsat 8 data can be collected from the [USGS earth explorer website](https://earthexplorer.usgs.gov/), while Sentinel 2 data can be collected from the [Copernicus Open Access Hub](https://scihub.copernicus.eu/dhus/#/home). However, both data sets can also be downloaded straight from the command line using their respective APIs.

### Sentinelsat



Note: landsatxplore has some compatibility issues with click - specifically, it needs click<8.0, but other geospatial programs require click 8.1.7. Best solution would be to run landsatxplore in a different environment.

landsatxplore was used to collect landsat data for this area. The following command was used: landsatxplore search --dataset landsat_ot_c2_l2 --location lat long --cloud 20
