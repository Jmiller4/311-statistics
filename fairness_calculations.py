import numpy as np
import pandas as pd
import csv
import datetime

count_syntax = ' - count'

class calculator:

    def __init__(self):
        self.key_metric_table = pd.DataFrame
        # key_metric_table should be a table with block groups as rows and request types as columns,
        # where index (i,j) is block group i's measure of the metric for requests of type j
        # an example of a "key metric" might be average wait time, for example
        self.demographics = pd.read_csv('demographics_table.csv', index_col=0)

    def load_table(self, path):
        self.key_metric_table = pd.read_csv(path, index_col=0)

    def get_avg_by_demo_and_request_type(self, demographic, request_type):

        '''
        details on what's going on here can be found here:
        https://www.notion.so/311-Statistics-Methodology-46037e1723914c4097db349fd0ca2c40
        '''

        key_metric_vector = self.key_metric_table[request_type]

        demographics_vector = self.demographics[demographic + count_syntax]

        raw_pop_sum = demographics_vector.sum()

        # we need to get rid of the block groups that haven't made a 311 call for the request type
        # importantly, we need to take those block groups out of both vectors
        # there might be a more efficient or cleaner way to do this

        for_cleaning = pd.concat([key_metric_vector,demographics_vector], axis=1)
        for_cleaning = for_cleaning.dropna(axis=0, subset=[request_type])
        pop_vec_cleaned = for_cleaning[demographic + count_syntax]
        metric_vec_cleaned = for_cleaning[request_type]

        s = pop_vec_cleaned.sum()

        if s == 0:
            return None, raw_pop_sum
        return pop_vec_cleaned.dot(metric_vec_cleaned) / s, raw_pop_sum


    def get_avg_across_demographics(self, demo_list, request_type):
        # this function finds the average response time for a specific request type across multiple *disjoint* demographics.
        # it could be useful to, say, find the average response time for everyone older than 50,
        # or to find the average response time for all non-white people.

        # importantly, the demographic categories need to be disjoint -- they either can all pertain to age and sex,
        # or all pertain to race and ethnicity, but not both. If you ask this function to tell you the average response time for
        # a certain request for white women, for example, it will give you the wrong answer.

        # the most up-do-date explanation of what's going on here can be found here:
        # https://www.notion.so/311-Statistics-Methodology-46037e1723914c4097db349fd0ca2c40

        sum = 0
        total_pop_sum = 0

        for demographic in demo_list:
            avg_time, raw_pop_sum = self.get_avg_by_demo_and_request_type(demographic, request_type)

            if avg_time is not None:
                sum += avg_time * raw_pop_sum
                total_pop_sum += raw_pop_sum

        if total_pop_sum == 0: return None
        return sum / total_pop_sum

def produce_tables(path_info, demo_path):
    # path_info should be a list of pairs, where
    #   - the first element is the path to the table containing the metric of interest per each block group and request type
    #   - the second element is the output path to write the new table to

    # demo_path should be a path to a csv file where for each line,
    #   - the first element is a human-readable name for a segment of the population
    #   - the rest of the elements are a series of DISJOINT demographic codes that describe that human-readable term

    # for each element of path_info (i.e. for each metric talked about in each table in path_info),
    # this function produces a table listing the average value of that metric for each demographic category defined in demo_path.


    request_frame = pd.read_csv('service_request_short_codes.csv')

    request_codes = request_frame.set_index('SR_SHORT_CODE').T.to_dict('records')[0]
    # request_codes is a dict with 311 short codes as keys and human-readable descriptions as values

    with open(demo_path) as f: #"broad_demographic_categories.csv"
        reader = csv.reader(f)
        demo_codes = {row[0]: row[1:] for row in reader}

    # demo_codes is a dict with human-readable demographic descriptions as keys and  lists of census codes as values


    # make a calculator object to do the calculations
    c = calculator()

    for x in path_info:

        c.load_table(x[0])

        # make a dataframe to put our results in
        frame = pd.DataFrame(columns=request_codes.values(), index=list(demo_codes.keys()))

        # populate the dataframe
        for request_code in list(request_codes.keys()):
            for demographic in list(demo_codes.keys()):
                avg_measure_of_metric = c.get_avg_across_demographics(demo_codes[demographic],request_code)

                if avg_measure_of_metric is None:
                    #this can happen if, among the block groups that have put out 311 calls of type request_code, no members of the demographic live in that block group.
                    # for example, no indigenous non-hispanic people live in any block group that has put out an "inacurate retail scales" 311 request.
                    frame.loc[demographic, request_codes[request_code]] = 'N/A'
                else:
                    #delta = datetime.timedelta(seconds=avg_time_in_sec)
                    frame.loc[demographic, request_codes[request_code]] = str(avg_measure_of_metric)

        # save the dataframe
        frame.to_csv(x[1])



# the below commented-out code is what I used to make the tables of average cue cuts per demographic/ request type.

# input_path_prefix = 'data preparation/average x by block group/bg_average_violations'
# output_path_prefix = 'avg_queue_cuts'
# bucket_types = ['1HR', '1SEC', '3HR', '6HR', '24HR']
# path_info = zip([input_path_prefix + x + '.csv' for x in bucket_types],[output_path_prefix + y + '.csv' for y in bucket_types])

# produce_tables(path_info, "broad_demographic_categories.csv")
