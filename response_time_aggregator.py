import pandas as pd
import numpy as np
import datetime as dt

'''
this code does two things:
1. it runs through every spreadsheet in the "311 data by type" folder and annotates each 311 request record with its response time. 
2. it makes a new spreadsheet populated with average response times per block group/ request type
'''



# read in a file to get the block group geo-ids

with open('cook_county_bg_geoids.txt') as f:
    geoids = [i[:-1] for i in f.readlines()] # [:-1] gets rid of the "\n at the end of each line"

# read in a file to get the request types
request_types = pd.read_csv('service_request_short_codes.csv')
sr_codes = [x[0] for x in request_types[['SR_SHORT_CODE']].values]

# initialize a new data frame. bg_averages[B, R] will contain the average response time for requests of type R where the location of the request is in block group B

bg_averages = pd.DataFrame(index=pd.Index(geoids,dtype=np.str),columns=pd.Index(sr_codes,dtype=np.str))


# this will be used to convert strings into datetime objects
dt_format = '%m/%d/%Y %I:%M:%S %p'

for code in sr_codes:

    print('starting', code)

    frame = pd.read_csv(r"311 data by type\311_"+code+".csv",
                     dtype={'CITY':np.str, 'STATE':np.str, 'ZIP_CODE':np.str,
                            'STREET_NUMBER':np.str, 'LEGACY_SR_NUMBER':np.str,
                            'PARENT_SR_NUMBER':np.str, 'SANITATION_DIVISION_DAYS':np.str,
                            'BLOCK_GROUP':np.str},
                     index_col=0)

    # get rid of rows with missing values for created or closed date
    frame = frame.dropna(axis=0, how='any', subset=['CREATED_DATE','CLOSED_DATE'])

    # get rid of duplicate records
    frame = frame[frame.DUPLICATE == False]

    # calculate time differences for all records

    frame['DELTA'] = frame.apply(lambda row:
                                 dt.datetime.strptime(row['CLOSED_DATE'], dt_format) -
                                 dt.datetime.strptime(row['CREATED_DATE'], dt_format), axis=1)

    # cast the timedeltas in seconds -- by default they're stored as nanoseconds, which leads to an overflow error
    # when we try to sum them up as we calculate the average.
    frame['DELTA'] = frame['DELTA'].dt.total_seconds()

    for bg in geoids:
        # make a new frame with just the requests from that block group
        new_frame = frame[frame.BLOCK_GROUP == bg]

        # get the total request time
        total = new_frame['DELTA'].sum()

        # calculate the average
        if len(new_frame) == 0:
            avg = None
        else:
            avg = total / len(new_frame)

        # set the value in the bg_averages dataframe
        bg_averages.loc[bg, code] = avg

    # save the data with appended deltas in a new spot.
    # im not overwriting the datasets on the off chance I messed something up here
    frame.to_csv('311 data by type with deltas/311_deltas_'+code+'.csv', index=False)

    print('finished data for ', code)


# save the averages
bg_averages.to_csv('bg_averages.csv')