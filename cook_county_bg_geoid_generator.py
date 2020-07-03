import shapefile

'''
This code will generate a file called 'cook_county_bg_geoids.txt', 
containing the geoids of every block group in cook county.

this code expects the most recent TIGER/Line shapefiles for Illinois 
(downloadable at https://www.census.gov/cgi-bin/geo/shapefiles/index.php)
to be unzipped in a directory called "shapefiles".
'''

cook_county_fp = '031'

# load shapefile
sf = shapefile.Reader("shapefiles/tl_2019_17_bg")
bg_data = sf.shapeRecords()

# open the file we're writing to
with open('cook_county_bg_geoids.txt', 'w') as f:
    for x in bg_data:
        if x.record.COUNTYFP == cook_county_fp:
            f.write(x.record.GEOID)
            f.write('\n')