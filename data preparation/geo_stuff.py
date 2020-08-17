import pandas as pd
import numpy as np
import csv

'''
just a file to find the maximum and minimum lattitudes and longitudes of all the requests in the dataset
it gives this output:

latitude:
	min:41.644567935
	max:42.022945905
longitude:
	min:-87.935172153
	max:-87.524498085

Latitude = North/South, with a higher number being more north
Longitude = East/West, with a higher number being more East

'''

request_types = pd.read_csv('service_request_short_codes.csv')
sr_codes = [x[0] for x in request_types[['SR_SHORT_CODE']].values]


lats = []
lons = []

for code in sr_codes:

    frame = pd.read_csv(f'311 data by request type/311_{code}.csv',
                        dtype={'CITY': np.str, 'STATE': np.str, 'ZIP_CODE': np.str,
                               'STREET_NUMBER': np.str, 'LEGACY_SR_NUMBER': np.str,
                               'PARENT_SR_NUMBER': np.str, 'SANITATION_DIVISION_DAYS': np.str,
                               'BLOCK_GROUP': np.str},
                        index_col=0)

    lats.append(frame['LATITUDE'].min())
    lats.append(frame['LATITUDE'].max())

    lons.append(frame['LONGITUDE'].min())
    lons.append(frame['LONGITUDE'].max())

print(f'latitude:\n\tmin:{min(lats)}\n\tmax:{max(lats)}')
print(f'longitude:\n\tmin:{min(lons)}\n\tmax:{max(lons)}')
