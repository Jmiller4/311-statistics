from csv import reader
import datetime
from tabulate import tabulate
from statistics import mean
from statistics import stdev
import os.path

def main():

    '''
    code to calculate average time between graffiti "registration (311 complaint)" & removal, by zip code, in Chicago.
    data set can be found at
    https://data.cityofchicago.org/Service-Requests/311-Service-Requests-Graffiti-Removal-Historical/hec5-y4x5/data
    with a download link at
    https://data.cityofchicago.org/api/views/hec5-y4x5/rows.csv?accessType=DOWNLOAD

    Note: this is historical data, thru 2018. Current dataset is at
    https://data.cityofchicago.org/Service-Requests/311-Service-Requests/v6vf-nfxy/data
    '''

    fname = '311_Service_Requests_-_Graffiti_Removal_-_Historical.csv'

    if not os.path.isfile(fname):
        print('dataset not found locally')
        exit()

    with open(fname, newline='') as dataset:

        data_reader = reader(dataset, delimiter=',')

        first_row = next(data_reader)
        creation_index = first_row.index('Creation Date')
        status_index = first_row.index('Status')
        completion_index = first_row.index('Completion Date')
        zip_index = first_row.index('ZIP Code')

        completion_stats = dict()
        other_stats = {'total_uncompleted': 0, 'total_completed': 0}

        for row in data_reader:

            zipcode = row[zip_index]
            if zipcode not in completion_stats.keys():
                completion_stats[zipcode] = {'uncompleted': 0, 'differences': []}

            if row[status_index] == 'Completed':

                creation_date = datetime.datetime.strptime(row[creation_index] , '%m/%d/%Y')
                completion_date = datetime.datetime.strptime(row[completion_index], '%m/%d/%Y')
                difference = completion_date - creation_date

                completion_stats[zipcode]['differences'].append(difference.days)
                other_stats['total_completed'] += 1
            else:
                completion_stats[zipcode]['uncompleted'] += 1
                other_stats['total_uncompleted'] += 1

        list_for_table = []
        for zipcode in completion_stats.keys():
            difference_list = completion_stats[zipcode]['differences']
            avg = mean(difference_list)
            if len(difference_list) > 1:
                std_deviation = stdev(difference_list)
            else:
                std_deviation = 'N/A'
            count = len(difference_list)
            uncompleted = completion_stats[zipcode]['uncompleted']

            list_for_table.append([zipcode, avg, std_deviation, count, uncompleted])

        nice_table = tabulate(list_for_table, headers=['Zip code', 'avg completion time', 'std dev of completion time',\
                                                '# of completed requests', '# of unfulfilled requests'])

        f = open('output.txt', 'w')
        f.write(nice_table)
        f.close()



main()

