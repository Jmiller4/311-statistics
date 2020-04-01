from csv import reader
from csv import writer
import datetime
from tabulate import tabulate
from statistics import mean
from statistics import stdev
import os.path

def main():

    '''
    code to calculate average time between the creation of graffitti complaints and the removal of said graffitti,
     by zip code, in Chicago.

    data set can be found at
    https://data.cityofchicago.org/Service-Requests/311-Service-Requests-Graffiti-Removal-Historical/hec5-y4x5/data
    with a download link at
    https://data.cityofchicago.org/api/views/hec5-y4x5/rows.csv?accessType=DOWNLOAD

    Note: this is historical data, thru 2018. Current dataset is at
    https://data.cityofchicago.org/Service-Requests/311-Service-Requests/v6vf-nfxy/data

    one interesting thing is, there are more than a few requests in here which are quite old --
    graffitti complaints registered in the 20th century (as early as the 1920s!) that only got covered up in the 2010s.
    To keep things relevant, I have a boolean called "ignore_old_requests" and a variable called "request_year_threshhold".
    If ignore_old_requests is True, then we only look at requests that were logged after request_year_threshhold.

    '''

    ignore_old_requests = True
    request_year_threshhold = 2005

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

            # get date request was filed, figure out if it's too old
            creation_date = datetime.datetime.strptime(row[creation_index], '%m/%d/%Y')
            if ignore_old_requests and creation_date.year < request_year_threshhold:
                continue

            # get the zip code
            zipcode = row[zip_index]

            # catch bad zips
            if len(zipcode) != 5:
                zipcode = 'Invalid'

            # make a new entry in our dictionary if we need to
            if zipcode not in completion_stats.keys():
                completion_stats[zipcode] = {'uncompleted': 0, 'differences': []}

            if row[status_index] == 'Completed':

                #get the time from the creation of the request to its completion, and update our dictionary

                completion_date = datetime.datetime.strptime(row[completion_index], '%m/%d/%Y')
                difference = completion_date - creation_date

                completion_stats[zipcode]['differences'].append(difference.days)
                other_stats['total_completed'] += 1
            else:
                completion_stats[zipcode]['uncompleted'] += 1
                other_stats['total_uncompleted'] += 1


        # calculate averages and standard deviations

        final_data = []
        for zipcode in completion_stats.keys():

            # ignore records with invalid zip codes
            if zipcode == 'Invalid':
                continue

            difference_list = completion_stats[zipcode]['differences']
            avg = mean(difference_list)
            if len(difference_list) > 1:
                std_deviation = stdev(difference_list)
            else:
                std_deviation = 'N/A'
            count = len(difference_list)
            uncompleted = completion_stats[zipcode]['uncompleted']

            final_data.append([zipcode, avg, std_deviation, count, uncompleted])

        final_data.sort(key=lambda x: x[1])
        final_data.reverse()

        headers = ['Zip code', 'avg completion time', 'std dev of completion time',\
                                                '# of completed requests', '# of unfulfilled requests']

        # write output as a nicely formatted table
        f = open('output.txt', 'w')
        f.write(tabulate(final_data, headers=headers))
        if ignore_old_requests:
            f.write('\n\nStats calculated ignoring all requests entered before ' + str(request_year_threshhold))
        f.close()

        # write output as csv
        with open('output.csv', 'w', newline='') as file:
            csv_writer = writer(file)
            csv_writer.writerow(headers)
            for row in final_data:
                csv_writer.writerow(row)

main()

