import pandas as pd
import json

'''
this code aggregates demographic data from json objects into one table, with block group geo-ids as rows and demographic stats as columns.
'''

# these variables are used to create column names
# better to have them in just one place so I don't risk typos
count_syntax = ' - count'
error_syntax = ' - error'

# this function creates column names from a list of codes
def columns_creator(l):
    columns = []
    for x in l:
        columns.extend([x + count_syntax, x + error_syntax])
    return columns


# read in a file to get the block group geo-ids
with open('cook_county_bg_geoids.txt') as f:
    geoids = [i[:-1] for i in f.readlines()] # [:-1] gets rid of the "\n at the end of each line"

# read in lists of codes for various demographic categories
age_sex_df = pd.read_csv('tables/code-descriptions-age-and-sex.csv')
race_eth_df = pd.read_csv('tables/code-descriptions-race-and-ethnicity.csv')

age_codes = age_sex_df['CODE'].to_list()
eth_codes = race_eth_df['CODE'].to_list()

# make the list of column names
age_columns = columns_creator(age_codes)
eth_columns = columns_creator(eth_codes)
all_columns = age_columns.extend(eth_columns)

# make the dataframe where wer're going to store our data
frame = pd.DataFrame(index=geoids, columns=all_columns)

# populate the dataframe
for g in geoids:

    #age and sex stats
    fname = 'tables/age-and-sex/' + g + '-B01001.json'
    with open(fname) as file:
        data = file.read()
    json_as_dict = json.loads(data)

    #index further into the dictionary
    important_stuff = json_as_dict['data']["15000US" + g]['B01001']

    #extract the stats
    for c in age_codes:
        frame.loc[g, c + error_syntax] = important_stuff['error'][c]
        frame.loc[g, c + count_syntax] = important_stuff['estimate'][c]

    # race and ethnicity stats -- same idea as above
    fname = 'tables/race-and-ethnicity/B03002/' + g + '-B03002.json'
    with open(fname) as file:
        data = file.read()
    json_as_dict = json.loads(data)
    important_stuff = json_as_dict['data']["15000US" + g]['B03002']
    for c in eth_codes:
        frame.loc[g, c + error_syntax] = important_stuff['error'][c]
        frame.loc[g, c + count_syntax] = important_stuff['estimate'][c]

#save the dataframe
frame.to_csv('demographics_table.csv')

#dataframe loc is row major
