import pandas as pd
import numpy as np

'''
this code makes a table with block group geo-ids as rows and request codes as columns,
such that entry (i,j) is the number of request of type j made from block group i
'''

def make_request_count_table():

    # read in a file to get the block group geo-ids
    with open('cook_county_bg_geoids.txt') as f:
        geoids = [i[:-1] for i in f.readlines()] # [:-1] gets rid of the "\n at the end of each line"

    # get the request type codes
    request_types = pd.read_csv('service_request_short_codes.csv')
    sr_codes = [x[0] for x in request_types[['SR_SHORT_CODE']].values]

    # make the dataframe where we're going to store our data
    frame = pd.DataFrame(data=0, index=geoids, columns=sr_codes, dtype=np.int)

    for code in sr_codes:
        request_data = pd.read_csv('311 data by request type/311_' + code + ".csv",
                            dtype={'CITY': np.str, 'STATE': np.str, 'ZIP_CODE': np.str,
                                   'STREET_NUMBER': np.str, 'LEGACY_SR_NUMBER': np.str,
                                   'PARENT_SR_NUMBER': np.str, 'SANITATION_DIVISION_DAYS': np.str,
                                   'BLOCK_GROUP': np.str},
                            index_col=0)

        # populate the dataframe with the data from this request type.
        frame[code] = request_data['BLOCK_GROUP'].value_counts()

    frame.fillna(0)
    frame.to_csv('request_count_table.csv', index=True)

make_request_count_table()