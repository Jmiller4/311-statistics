import pandas as pd
import numpy as np
import datetime as dt
import math
import time

def append_bucket_violations(df, bucket_col, new_col):
    # time complexity: length of list * number of buckets
    d = {}
    for row in df.index:
        b = df.loc[row, bucket_col]
        if b not in d.keys():
            d[b] = 1
        else:
            d[b] += 1
        violations = 0
        # for each iteration (over the rows), d[k] gives the number of items in bucket k that we've already seen.
        # so if item i has bucket value b < k, and d[k] has been initialized, then d[k] items (from bucket k) unfairly went before item i.
        # do this with all buckets k, and you get the total number of items that unfairly went before item i.
        for k in d.keys():
            if b < k:
                violations += d[k]
        df.loc[row, new_col] = violations
    return df



# this will be used to convert strings into datetime objects
dt_format = '%m/%d/%Y %I:%M:%S %p'

early_date = dt.datetime(2018,6,20,0,0,0,0)
# the earliest request in the dataset was made on july 1st 2018, so this date is before every request in the dataset.

def append_buckets():

    ''' this function appends datasets with created date in seconds and closed date in seconds (measured from a hand-picked date from before the dataset began)'''

    request_types = pd.read_csv('service_request_short_codes.csv')
    sr_codes = [x[0] for x in request_types[['SR_SHORT_CODE']].values]

    for code in sr_codes:

        frame = pd.read_csv(r"311 data by type with deltas\311_deltas_" + code + ".csv",
                            dtype={'CITY': np.str, 'STATE': np.str, 'ZIP_CODE': np.str,
                                   'STREET_NUMBER': np.str, 'LEGACY_SR_NUMBER': np.str,
                                   'PARENT_SR_NUMBER': np.str, 'SANITATION_DIVISION_DAYS': np.str,
                                   'BLOCK_GROUP': np.str},
                            index_col=None)

        frame['CREATED_DATE_SEC'] = frame.apply(lambda row: (dt.datetime.strptime(row['CREATED_DATE'],dt_format) - early_date).total_seconds(), axis=1)

        frame['CLOSED_DATE_SEC'] = frame.apply(
            lambda row: (dt.datetime.strptime(row['CLOSED_DATE'], dt_format) - early_date).total_seconds(), axis=1)

        frame['CR_BUCKET_1HR']  = frame.apply(lambda row: math.floor(row['CREATED_DATE_SEC'] / 3600),      axis=1)
        frame['CR_BUCKET_3HR']  = frame.apply(lambda row: math.floor(row['CREATED_DATE_SEC'] /(3600 * 3)),  axis=1)
        frame['CR_BUCKET_6HR']  = frame.apply(lambda row: math.floor(row['CREATED_DATE_SEC'] /(3600 * 6)),  axis=1)
        frame['CR_BUCKET_24HR'] = frame.apply(lambda row: math.floor(row['CREATED_DATE_SEC'] /(3600 * 24)), axis=1)

        frame.to_csv(r"311 data buckets\311_buckets_" + code + ".csv", index=False)

def calculate_fifo_violations_for_all_datasets():

    request_types = pd.read_csv('service_request_short_codes.csv')
    sr_codes = [x[0] for x in request_types[['SR_SHORT_CODE']].values]

    for code in sr_codes:

        print(code)

        frame = pd.read_csv(r"311 data buckets\311_buckets_" + code + ".csv",
                            dtype={'CITY': np.str, 'STATE': np.str, 'ZIP_CODE': np.str,
                                   'STREET_NUMBER': np.str, 'LEGACY_SR_NUMBER': np.str,
                                   'PARENT_SR_NUMBER': np.str, 'SANITATION_DIVISION_DAYS': np.str,
                                   'BLOCK_GROUP': np.str},
                            index_col=None)

        frame = frame.sort_values(by=['CLOSED_DATE_SEC'])

        bucket_lengths = ['1HR', '3HR', '6HR', '24HR']

        for l in bucket_lengths:
            frame = append_bucket_violations(frame, 'CR_BUCKET_'+l, 'CR_BUCKET_'+l+'_WAIT_TIME')

        frame = append_bucket_violations(frame, 'CREATED_DATE_SEC', 'CR_BUCKET_1SEC_WAIT_TIME')

        frame.to_csv(r"311 data buckets\311_buckets_" + code + ".csv", index=False)


# calculate_fifo_violations_for_all_datasets()



'''
What follows is UNUSED code to find the number of inversions in a list.
pretty much a slight modification of mergesort
i wrote this thinking it might faster than the other way of counting fifo violations
due to mergesort's n * log n runtime.
HOWEVER, it actually ran much slower, on my machine at least.
'''
def count_inversions(l, frame, bucket_col, inv_col):
    split_point = len(l)//2
    frame[inv_col] = [0] * len(frame)
    list, inversions, frame = count_inversions_helper(l[:split_point], l[split_point:], frame, bucket_col, inv_col)
    return inversions, frame

def count_inversions_helper(left, right, frame, bucket_col, inv_col):

    l_inversions, r_inversions = 0, 0

    if len(left) > 1:
        left_split = len(left)//2
        left, l_inversions, frame = count_inversions_helper(left[:left_split], left[left_split:], frame, bucket_col, inv_col)

    if len(right) > 1:
        right_split = len(right) // 2
        right, r_inversions, frame = count_inversions_helper(right[:right_split], right[right_split:], frame, bucket_col, inv_col)

    merged_list = []
    inversions = l_inversions + r_inversions
    while len(left) > 0 and len(right) > 0:
        if frame.loc[left[0], bucket_col] <= frame.loc[right[0], bucket_col]:
            merged_list.append(left.pop(0))
        else:
            frame.loc[right[0], inv_col] += len(left)
            inversions += len(left)
            merged_list.append(right.pop(0))


    if len(left) == 0:
        merged_list.extend(right)
    else:
        merged_list.extend(left)
    return merged_list, inversions, frame


