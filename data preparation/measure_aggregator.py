import pandas as pd
import numpy as np
import datetime as dt

def make_averages_table(column, path_prefix, output_path):
    '''
    this file takes in a column and information about file paths
    it finds the average value of the column for all (request type, block group) pairs,
    and saves that information to a new dataframe
    '''

    # read in a file to get the block group geo-ids
    with open('cook_county_bg_geoids.txt') as f:
        geoids = [i[:-1] for i in f.readlines()] # [:-1] gets rid of the "\n at the end of each line"

    # read in a file to get the request types
    request_types = pd.read_csv('service_request_short_codes.csv')
    sr_codes = [x[0] for x in request_types[['SR_SHORT_CODE']].values]

    # initialize a new data frame. averages[B, R] will contain the average metric (defined by the input column to look at) for requests of type R where the location of the request is in block group B

    averages = pd.DataFrame(index=pd.Index(geoids,dtype=np.str),columns=pd.Index(sr_codes,dtype=np.str))

    for code in sr_codes:

        print('starting ', code)

        frame = pd.read_csv(path_prefix + code+".csv",
                         dtype={'CITY':np.str, 'STATE':np.str, 'ZIP_CODE':np.str,
                                'STREET_NUMBER':np.str, 'LEGACY_SR_NUMBER':np.str,
                                'PARENT_SR_NUMBER':np.str, 'SANITATION_DIVISION_DAYS':np.str,
                                'BLOCK_GROUP':np.str},
                         index_col=0)


        # make new frames with just the requests from each block group
        filtered_frames = {}
        for bg in geoids:
            filtered_frames[bg] = frame[frame.BLOCK_GROUP == bg]

        for bg in geoids:

            f = filtered_frames[bg]

            #find the average of the given metric for the block group, and put that in the dataframe

            #it could be the case that no requests for the given type ever came out of the block group,
            #so check for that first
            if len(f) == 0:
                avg = None
            else:
                # get the total request time
                total = f[column].sum()
                # calculate the average
                if total == 0:
                    avg = 0
                else:
                    avg = total / len(f)

            # set the value in the "averages" dataframe
            averages.loc[bg, code] = avg

    # save the averages
    averages.to_csv(output_path)

def make_tables_for_bucket_fifo_violations():

    bucket_lengths = ['1HR', '3HR', '6HR', '24HR']

    for l in bucket_lengths:
        make_averages_table('CR_BUCKET_'+l, '311 data buckets/311_buckets_', 'bg_average_violations' + l + '.csv')

    make_averages_table('CREATED_DATE_SEC',  '311 data buckets/311_buckets_', 'bg_average_violations1SEC.csv')


'''
this is a function i used to test some stuff out.
it's not important.
'''
def bucket_cut_test():

    # pretty much, I wrote this function to check that the average "cuts" in line for each block group/ request type pair
    # are non-increasing as we make the buckets less and less granular
    # if it printed anything out, that would've been a problem, but it didn't

    bucket_lengths = ['1SEC','1HR', '3HR', '6HR', '24HR']
    # read in a file to get the block group geo-ids
    with open('cook_county_bg_geoids.txt') as f:
        geoids = [i[:-1] for i in f.readlines()] # [:-1] gets rid of the "\n at the end of each line"

    # read in a file to get the request types
    request_types = pd.read_csv('service_request_short_codes.csv')
    sr_codes = [x[0] for x in request_types[['SR_SHORT_CODE']].values]

    dfs = {}
    for x in bucket_lengths:
        dfs[x] = pd.read_csv('average x by block group/bg_average_violations' + x + '.csv', index_col=0)

    # test this out with the files that list average wait times
    for g in geoids:
        g = int(g)
        for c in sr_codes:
            if np.isnan(dfs['1SEC'].loc[g, c]):
                # make sure it's a NaN everywhere else
                for l in bucket_lengths:
                    if not np.isnan(dfs[l].loc[g,c]):
                        print('geoid: ', g, '\ncode: ', c, '\nbucket type: ', l, 'is not NaN even though 1sec version is')
            else:
                val = dfs['1SEC'].loc[g,c]
                for l in bucket_lengths:
                    new_val = dfs[l].loc[g,c]
                    #as the buckets get bigger, the number of cuts should decrease or stay the same
                    if not new_val <= val:
                        print('geoid: ', g, '\ncode: ', c, '\nbucket type: ', l, 'increases from smaller bucket size')
                    val = new_val

    # test this out for each individual record
    wait_time_columns = ['CR_BUCKET_' + x + '_WAIT_TIME' for x in bucket_lengths]
    for c in sr_codes:
        df = pd.read_csv('311 data buckets/311_buckets_' + c + '.csv', index_col=0)
        for x in df.index:
            val = df.loc[x, 'CR_BUCKET_1SEC_WAIT_TIME']
            for col in wait_time_columns:
                new_val = df.loc[x, col]
                # as the buckets get bigger, the number of cuts should decrease or stay the same
                if not new_val <= val:
                    print('code: ', c, '\nsr number: ', x, '\nbucket type: ', col, 'increases from smaller bucket size')
                val = new_val


'''
what follows is an early pass at make_averages_table() function.
i wrote it when I was just interested in calculating the average raw wait time per block group,
so this code kind of does two things at once.
also not important. keeping it for posterity?
'''
def first_function():
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
    bg_averages.to_csv('bg_averages_raw_time.csv')