# Singapore NDVI Project

This project investigates the use of cloud masks to remove clouds while calculating NDVI over time.

Note: landsatxplore has some compatibility issues with click - specifically, it needs click<8.0, but other geospatial programs require click 8.1.7. Best solution would be to run landsatxplore in a different environment.

landsatxplore was used to collect landsat data for this area. The following command was used: landsatxplore search --dataset landsat_ot_c2_l2 --location lat long --cloud 20
