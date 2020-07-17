import pandas as pd
import numpy as np
import datetime as dt
import math

# this will be used to convert strings into datetime objects
dt_format = '%m/%d/%Y %I:%M:%S %p'
#
early_date = dt.datetime(2018,6,20,0,0,0,0)
# the earliest request in the dataset was made on july 1st 2018, so this date is before every request in the dataset.
# a bucket number is the number of full (1SEC, 1HR, 3HR, 6HR, 24HR) periods since this date/time.

def append_buckets():

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

def calculate_displacement_for_rows(df, bucket_col, new_col):
    vc = df[bucket_col].value_counts()
    vc = vc.sort_index()
    vcs = vc.cumsum()

    acceptable = {}
    lower = 0
    for x in vcs.index:
        acceptable[x] = {}
        acceptable[x]['start'] = lower
        acceptable[x]['end'] = vcs[x]
        lower = vcs[x] + 1

    df['acc_start'] = df.apply(lambda row: acceptable[row[bucket_col]]['start'], axis=1)
    df['acc_end'] = df.apply(lambda row: acceptable[row[bucket_col]]['end'], axis=1)

    df[new_col] = df.apply(lambda row: (row['ORDER'] - row['acc_end']) * (row['ORDER'] > row['acc_end'])
                                              + (row['ORDER'] - row['acc_start']) * (row['ORDER'] < row['acc_start'])
                                  , axis=1)

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
            frame = calculate_displacement_for_rows(frame, 'CR_BUCKET_' + b_size, 'CR_BUCKET_' + b_size + '_DISPLACEMENT')

        frame.to_csv(r"311 data by request type\311_" + code + ".csv", index=False)

append_datasets_with_displacements()