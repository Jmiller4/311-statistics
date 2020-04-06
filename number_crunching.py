from csv import DictReader
from csv import writer
import datetime
from tabulate import tabulate
from statistics import mean
from statistics import stdev
import os.path

class Zip:

    '''
    class for holding all records pertaining to a given zip code,
    and doing calculations on said records
    '''

    def __init__(self, code):
        self.code = code
        self.records = []
        self.response_times = []
        self.completed_requests = 0
        self.uncompleted_requests = 0
        self.total_requests = 0
        self.avg_response_time = 0
        self.stdev_of_response_times = 0

    def add_record(self, creation_date, completion_date, status, historical):
        if not historical:
            creation_date = creation_date[:10]
            completion_date = completion_date[:10]
        self.records.append([creation_date, completion_date, status])
        self.total_requests += 1

    def compute_stats(self, ignore_old_requests=False, request_year_threshold=2005):

        self.completed_requests = 0
        self.uncompleted_requests = 0
        self.response_times = []

        for rec in self.records:

            creation_date = datetime.datetime.strptime(rec[0], '%m/%d/%Y')

            if ignore_old_requests and creation_date.year < request_year_threshold:
                continue

            if rec[2] != "Completed":
                self.uncompleted_requests += 1
                continue

            self.completed_requests += 1
            completion_date = datetime.datetime.strptime(rec[1], '%m/%d/%Y')
            response_time = completion_date - creation_date
            self.response_times.append(response_time.days)

        if len(self.response_times) > 0:
            self.avg_response_time = mean(self.response_times)
        else:
            self.avg_response_time = 'N/A'

        if len(self.response_times) > 1:
            self.stdev_of_response_times = stdev(self.response_times)
        else:
            self.stdev_of_response_times = 'N/A'

def main():

    '''
    code to calculate average time between the creation of graffitti complaints and the removal of said graffitti,
     by zip code, in Chicago.

    historical data set can be found at
    https://data.cityofchicago.org/Service-Requests/311-Service-Requests-Graffiti-Removal-Historical/hec5-y4x5/data
    with a download link at
    https://data.cityofchicago.org/api/views/hec5-y4x5/rows.csv?accessType=DOWNLOAD

    Current dataset is at
    https://data.cityofchicago.org/Service-Requests/311-Service-Requests/v6vf-nfxy/data

    one interesting thing is, there are more than a few requests in the historical data which are quite old --
    graffitti complaints registered in the 20th century (as early as the 1920s!) that only got covered up in the 2010s.
    To keep things relevant, I have a boolean called "ignore_old_requests" and a variable called "request_year_threshhold".
    If ignore_old_requests is True, then we only look at requests that were logged after request_year_threshhold.

    '''

    ignore_old_requests = True
    request_year_threshhold = 2005
    use_historical_data = True
    use_current_data = True

    historical_fname = '311_Service_Requests_-_Graffiti_Removal_-_Historical.csv'
    current_fname = '311_Service_Requests.csv'

    if not use_current_data and not use_current_data:
        exit()

    if not os.path.isfile(historical_fname):
        print('historical dataset not found locally')
        exit()

    if not os.path.isfile(current_fname):
        print('current dataset not found locally')
        exit()

    zips = dict()

    fnames = [historical_fname, current_fname]

    # The two datasets use different header names.
    # These lists help us pass over both datasets while adjusting for that difference.
    zip_headers = ['ZIP Code', 'ZIP_CODE']
    creation_headers = ['Creation Date', 'CREATED_DATE']
    completion_headers = ['Completion Date', 'CLOSED_DATE']
    status_headers = ['Status', 'STATUS']

    file_list = []
    if use_historical_data: file_list.append(0)
    if use_current_data: file_list.append(1)

    for i in [0,1]:
        with open(fnames[i], newline='') as dataset:
            data_reader = DictReader(dataset, delimiter=',')
            for row in data_reader:

                # skip entries from the current dataset that aren't graffitti removal requests,
                # and also skip legacy requests -- ostensibly those are in the other dataset as well
                # also skip duplicate requests
                if i == 1 and (row['SR_SHORT_CODE'] != 'GRAF' or row['LEGACY_RECORD'] == 'true'
                               or row['DUPLICATE'] == 'true') : continue

                # get the zip code
                zipcode = row[zip_headers[i]]

                # catch bad zips
                if not is_zipcode(zipcode):
                    zipcode = 'Invalid'

                # make a new entry in our dictionary if we need to
                if zipcode not in zips.keys():
                    zips[zipcode] = Zip(zipcode)

                # add the record to the appropriate Zip object
                zips[zipcode].add_record(row[creation_headers[i]],row[completion_headers[i]],
                                         row[status_headers[i]], i == 0)

    # get averages and standard deviations

    final_data = []
    for zipcode in zips.keys():

        #ignore records with invalid zip codes

        if zipcode == 'Invalid': continue

        zip_obj = zips[zipcode]

        zip_obj.compute_stats(ignore_old_requests, request_year_threshhold)

        final_data.append([zipcode, zip_obj.avg_response_time, zip_obj.stdev_of_response_times,
                           zip_obj.completed_requests, zip_obj.uncompleted_requests])

    final_data.sort(key=lambda x: x[1])
    final_data.reverse()

    headers = ['Zip code', 'avg completion time', 'std dev of completion time',
                                            '# of completed requests', '# of unfulfilled requests']

    output_fname = 'output'
    if use_historical_data and not use_current_data:
        output_fname += "_historical"
    elif use_current_data and not use_historical_data:
        output_fname += "_current"
    else:
        output_fname += "_all"
    if ignore_old_requests and use_historical_data:
        output_fname += "_past_" + str(request_year_threshhold)

    output_txt_fname = output_fname + ".txt"
    output_csv_fname = output_fname + ".csv"

    # write output as a nicely formatted table
    f = open(output_txt_fname, 'w')
    f.write(tabulate(final_data, headers=headers))
    if ignore_old_requests:
        f.write('\n\nStats calculated ignoring all requests entered before ' + str(request_year_threshhold))
    f.close()

    # write output as csv
    with open(output_csv_fname, 'w', newline='') as file:
        csv_writer = writer(file)
        csv_writer.writerow(headers)
        for row in final_data:
            csv_writer.writerow(row)


def is_zipcode(string):
    return len(string) == 5 and all([c in [str(y) for y in range(10)] for c in string])

main()

