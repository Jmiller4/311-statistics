import pandas as pd
import numpy as np
import datetime as dt
import math
import time
import matplotlib.pyplot as plt
import matplotlib.dates as dates

# this will be used to convert strings into datetime objects
dt_format = '%m/%d/%Y %I:%M:%S %p'
#
early_date = dt.datetime(2018,6,20,0,0,0,0)
# the earliest request in the dataset was made on july 1st 2018, so this date is before every request in the dataset.
# a bucket number is the number of full (1SEC, 1HR, 3HR, 6HR, 24HR) periods since this date/time.

def append_bucket_columns():

    ''' this function appends datasets with created date in seconds and closed date in seconds (measured from a hand-picked date from before the dataset began).
    it also datasets with their "bucket numbers" for other bucket sizes.
    '''

    request_types = pd.read_csv('service_request_short_codes.csv')
    sr_codes = [x[0] for x in request_types[['SR_SHORT_CODE']].values]

    for code in sr_codes:

        frame = pd.read_csv(r"311 data by type with deltas\311_deltas_" + code + ".csv",
                            dtype={'CITY': np.str, 'STATE': np.str, 'ZIP_CODE': np.str,
                                   'STREET_NUMBER': np.str, 'LEGACY_SR_NUMBER': np.str,
                                   'PARENT_SR_NUMBER': np.str, 'SANITATION_DIVISION_DAYS': np.str,
                                   'BLOCK_GROUP': np.str},
                            index_col=None)

        frame['CR_BUCKET_1SEC'] = frame.apply(lambda row: (dt.datetime.strptime(row['CREATED_DATE'],dt_format) - early_date).total_seconds(), axis=1)

        frame['CLOSED_DATE_SEC'] = frame.apply(
            lambda row: (dt.datetime.strptime(row['CLOSED_DATE'], dt_format) - early_date).total_seconds(), axis=1)

        frame['CR_BUCKET_1HR']  = frame.apply(lambda row: math.floor(row['CR_BUCKET_1SEC'] / 3600),      axis=1)
        frame['CR_BUCKET_3HR']  = frame.apply(lambda row: math.floor(row['CR_BUCKET_1SEC'] /(3600 * 3)),  axis=1)
        frame['CR_BUCKET_6HR']  = frame.apply(lambda row: math.floor(row['CR_BUCKET_1SEC'] /(3600 * 6)),  axis=1)
        frame['CR_BUCKET_24HR'] = frame.apply(lambda row: math.floor(row['CR_BUCKET_1SEC'] /(3600 * 24)), axis=1)

        frame.to_csv(r"311 data buckets\311_buckets_" + code + ".csv", index=False)

def calculate_displacement_for_all_rows(df, bucket_col, new_col):

    # here's how this works.
    # for each bucket, we calculate the part of the output queue that requests from that bucket would occupy.
    # for example, if we're working with 1-hour buckets and 5 requests come in 12-1PM and 10 requests come in from 1-2PM,
    # we'd expect to see the 12-1PM requests completed 0-4th, and the 1-2PM requests completed 4-15th.

    # then we compare each request's completion spot (stored in the "ORDER" column)
    # with the "acceptable range" for requests in that bucket.
    # if the request is completed inside the range, it is given a displacement of 0
    # if the request is completed after the end of the range, it is given a displacement of (request's spot in queue) - (the end of the range)
    # if the request is completed before the beginning of the range, its displacement is (request's spot in queue) - (the beginning of the range)
    # so late requests get positive displacement and early requests get negative displacement
    # keeping with the above example, if a 12-1PM request was completed 11th, it would get a displacement of 11 - 4 = 7
    # and if a 1-2PM request was completed 2nd, it would get a displacement of 2 - 5 = -3.


    # first, get the number of requests in each bucket. vc is a pandas series where each index is a bucket number and the actual element
    # is the number of requests in that bucket
    vc = df[bucket_col].value_counts()

    # calculate the acceptable range for each bucket
    # there is probably a way to do this more quickly...
    vc = vc.sort_index()
    vcs = vc.cumsum()

    acceptable = {}
    lower = 0
    for x in vcs.index:
        acceptable[x] = {}
        acceptable[x]['start'] = lower
        acceptable[x]['end'] = vcs[x] - 1
        lower = vcs[x]

    # appending every request with that request's bucket's acceptable range seems weird,
    # but this runs faster than some other things I tried.
    df['acc_start'] = df.apply(lambda row: acceptable[row[bucket_col]]['start'], axis=1)
    df['acc_end'] = df.apply(lambda row: acceptable[row[bucket_col]]['end'], axis=1)

    # this is where the displacement is set, according to the logic discussed above
    # by multiplying different parts of an expression by different boolean values, we're essentially mimicking branches
    df[new_col] = df.apply(lambda row: (row['ORDER'] - row['acc_end']) * (row['ORDER'] > row['acc_end'])
                                              + (row['ORDER'] - row['acc_start']) * (row['ORDER'] < row['acc_start'])
                                  , axis=1)

    # get rid of the temporary columns we added on
    df.drop(columns=['acc_start', 'acc_end'], inplace=True)

    return df

def append_datasets_with_displacements():

    request_types = pd.read_csv('service_request_short_codes.csv')
    sr_codes = [x[0] for x in request_types[['SR_SHORT_CODE']].values]

    for code in sr_codes:
        print(code)

        frame = pd.read_csv(r"311 data by request type\311_" + code + ".csv", index_col=None, dtype={'CITY':np.str, 'STATE':np.str, 'ZIP_CODE':np.str, 'STREET_NUMBER':np.str, 'LEGACY_SR_NUMBER':np.str, 'PARENT_SR_NUMBER':np.str, 'SANITATION_DIVISION_DAYS':np.str})

        frame.sort_values(by=['CLOSED_DATE_SEC'],inplace=True)
        frame.reset_index(drop=True, inplace=True)
        frame['ORDER'] = frame.index

        for b_size in ['1SEC','1HR','3HR','6HR','24HR']:
            frame = calculate_displacement_for_all_rows(frame, 'CR_BUCKET_' + b_size, 'CR_BUCKET_' + b_size + '_DISPLACEMENT')

        frame.to_csv(r"311 data by request type\311_" + code + ".csv", index=False)

# append_datasets_with_displacements()


def find_avg_displacement_across_demos():
    request_types = pd.read_csv('service_request_short_codes.csv')
    sr_codes = [x[0] for x in request_types[['SR_SHORT_CODE']].values]

    b_sizes = ['1SEC','1HR','3HR','6HR','24HR']

    new_frame = pd.DataFrame(index=sr_codes, columns=b_sizes)

    for code in sr_codes:
        print(code)
        code_frame = pd.read_csv(r"311 data by request type\311_" + code + ".csv", index_col=None, dtype={'CITY':np.str, 'STATE':np.str, 'ZIP_CODE':np.str, 'STREET_NUMBER':np.str, 'LEGACY_SR_NUMBER':np.str, 'PARENT_SR_NUMBER':np.str, 'SANITATION_DIVISION_DAYS':np.str})

        for size in b_sizes:

            new_frame.loc[code, size] = code_frame.apply(lambda row: abs(row['CR_BUCKET_' + size + '_DISPLACEMENT']), axis=1).sum() / len(code_frame)

    new_frame.to_csv('average_displacement.csv')

#find_avg_displacement_across_demos()

# timestamps = pd.date_range(start="2017-05-08", freq="10T", periods=6*6)
# timedeltas = timestamps - pd.to_datetime("2017-05-08")
# yy = np.random.random((len(timedeltas),10))
# df = pd.DataFrame(data=yy, index=timedeltas)
#
# fig,axes=plt.subplots()
# axes.plot_date(df.index, df.values, '-')
#
# axes.xaxis.set_major_locator(dates.HourLocator(byhour=range(0,24,2)))
# axes.xaxis.set_minor_locator(dates.MinuteLocator(byminute=range(0,24*60,10)))
# axes.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
#
# plt.show()
#
# exit()


