import pandas as pd
import numpy as np
from bg_decider import bg_decider

''' this code takes chicago's 311 dataset and appends each record with its appropriate block group geo-id (all records have a location, specified with coordinates), then saves the result as a new file.'''


'''the dataset is available at:
https://data.cityofchicago.org/Service-Requests/311-Service-Requests/v6vf-nfxy
I recommend you filter out 311 info-only complaints and aircraft noise complaints before downloading -- this will halve the size of the dataset, and both of those request types are marked as fulfilled immediately, making them irrelevant for our purposes.
'''

'''also -- there are a few points in the dataset that aren't in a block group. they're all on the border with indiana.'''

od = pd.read_csv("311_Service_Requests.csv",dtype={'CITY':np.str, 'STATE':np.str, 'ZIP_CODE':np.str, 'STREET_NUMBER':np.str, 'LEGACY_SR_NUMBER':np.str, 'PARENT_SR_NUMBER':np.str, 'SANITATION_DIVISION_DAYS':np.str})

bd = bg_decider()

od["BLOCK_GROUP"] = od.apply(lambda row: bd.decide_bg(row.LATITUDE, row.LONGITUDE), axis=1)

od.to_csv('311_Serviee_Requests_bg', index=False)
