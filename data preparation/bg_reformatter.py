import pandas as pd

''' 
pandas auto-interpreted the block group geo-ids as floats and accordingly put a '.0' after them. this code undoes that. 
'''

codes_df = pd.read_csv('service_request_short_codes.csv')

print(codes_df['SR_SHORT_CODE'])

for code in codes_df['SR_SHORT_CODE']:
    df = pd.read_csv(r'311 data by type\311_'+code+'.csv')
    df['BLOCK_GROUP'] = df.apply(lambda row: str(row.BLOCK_GROUP)[:-2], axis=1)
    df.to_csv(r'311 data by type\311_'+code+'.csv')