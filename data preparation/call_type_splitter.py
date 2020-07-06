import pandas as pd
import numpy as np

'''
this code does two things.
1. it splits the 311 dataset up into multiple smaller sets, one for each type of service request
2. it saves a small csv file associating human-readable service request types with their short codes
'''

df = pd.read_csv("311_Service_Requests_bg.csv",dtype={'CITY':np.str, 'STATE':np.str, 'ZIP_CODE':np.str, 'STREET_NUMBER':np.str, 'LEGACY_SR_NUMBER':np.str, 'PARENT_SR_NUMBER':np.str, 'SANITATION_DIVISION_DAYS':np.str})

#get a list of unique request types
unique_sr_types = np.unique(df[['SR_TYPE']].values)

#create the mapping between request types and short codes
mapping = pd.DataFrame(columns=['SR_SHORT_CODE','SR_TYPE'])

for v in unique_sr_types:

    #make a new dataframe
    df_v = df[df['SR_TYPE']==v]

    # re-write the indices of the new dataframe so they start from 0 and go in order
    df_v.index = range(len(df_v))

    # index into any row for the short code associated with this service request
    sr_short_code = (df_v.loc[1,'SR_SHORT_CODE'])

    # save the new dataframe as a new csv
    df_v.to_csv(r'311 data by type\311_' + sr_short_code + '.csv', index=False)

    # update our mapping
    mapping = mapping.append({'SR_SHORT_CODE':sr_short_code, 'SR_TYPE':v}, ignore_index=True)

# save the mapping
mapping.to_csv('service_request_short_codes.csv', index=False)