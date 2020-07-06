import numpy as np
import pandas as pd
import csv
import datetime

count_syntax = ' - count'

class calculator:

    def __init__(self):
        self.response_times = pd.read_csv('bg_averages.csv', index_col=0)
        self.demographics = pd.read_csv('demographics_table.csv', index_col=0)



    def get_avg_by_demo_and_request_type(self, demographic, request_type, returnmode='avg'):

        ''' this function takes in a demographic a request type, and estimates the average time a member
        of that demographic would wait to have a 311 request of that type fulfilled.

        This is calculated as
        P(someone lives in block group 1 | they're in that demographic) * The avg response time for that request in that block group + ...
        summing over all block groups.

        P(someone lives in block group x | they're in that demographic) = # of people of that demographic in block group x / total people of that demographic in the city.

        And of course, that denominator can be drawn out of the sum bc its the same across all terms.

        Things get slightly more complicated when you consider that some block groups have never made a 311 call for certain types of requests
        -- meaning the average response time for those block groups is undefined.

                                                                                                                                                                                                       So those block groups get left out of the sum, and instead of dividing by the total # of people in the city of that demographic, we
        instead divide by the total # of people in the city of that demographic *who live in block groups that have made calls about that specific 311 request*.
        '''


        response_times_vector = self.response_times[request_type]

        demographics_vector = self.demographics[demographic + count_syntax]

        raw_pop_sum = demographics_vector.sum()

        # we need to get rid of the block groups that haven't made a 311 call for the request type
        # importantly, we need to take those block groups out of both vectors
        # there might be a more efficient or cleaner way to do this

        for_cleaning = pd.concat([response_times_vector,demographics_vector], axis=1)
        for_cleaning = for_cleaning.dropna(axis=0, subset=[request_type])
        pop_vec_cleaned = for_cleaning[demographic + count_syntax]
        resp_vec_cleaned = for_cleaning[request_type]

        s = pop_vec_cleaned.sum()

        if returnmode == 'avg':
            if s == 0: return None
            return(pop_vec_cleaned.dot(resp_vec_cleaned) / s)
        elif returnmode == 'detail':
            if s == 0: return None, raw_pop_sum
            return(pop_vec_cleaned.dot(resp_vec_cleaned) / s, raw_pop_sum)


        # I used the below code to double-check that all the  pandas functions were working as intended
        # with open('cook_county_bg_geoids.txt') as f:
        #     geoids = [int(i[:-1]) for i in f.readlines()]
        # dot_prod_sum = 0
        # pop_sum = 0
        # for g in geoids:
        #     if not np.isnan(r[g]):
        #         dot_prod_sum += r[g] * d[g]
        #         pop_sum += d[g]
        # print(dot_prod_sum / pop_sum)
        #

    def get_avg_across_demographics(self, demo_list, request_type):
        # this function finds the average response time for a specific request type across multiple *disjoint* demographics.
        # it could be useful to, say, find the average response time for everyone older than 50,
        # or to find the average response time for all non-white people.

        # importantly, the demographic categories need to be disjoint -- they either can all pertain to age and sex,
        # or all pertain to race and ethnicity, but not both. If you ask this function to tell you the average response time for
        # a certain request for white women, for example, it will give you the wrong answer.

        # this function works by calculating
        # P(you're in demographic x | you're in a demographic in demo_list) * avg response time for the request for people in demographic x + ...,
        # summing over all x in demo_list.

        # P(you're in demographic x | you're in a demographic in demo_list) is calculated as
        # number of people in demographic x in the whole city / number of people in any demographic in demo_list in the whole city

        sum = 0
        total_pop_sum = 0

        for demographic in demo_list:
            avg_time, raw_pop_sum = self.get_avg_by_demo_and_request_type(demographic, request_type, returnmode='detail')

            if avg_time is not None:
                sum += avg_time * raw_pop_sum
                total_pop_sum += raw_pop_sum

        if sum == 0: return None
        return sum / total_pop_sum

def produce_table():
    request_frame = pd.read_csv('service_request_short_codes.csv')

    request_codes = request_frame.set_index('SR_SHORT_CODE').T.to_dict('records')[0]
    # request_codes is a dict with 311 short codes as keys and human-readable descriptions as values

    with open("broad_demographic_categories.csv") as f:
        reader = csv.reader(f)
        demo_codes = {row[0]: row[1:] for row in reader}

    # demo_codes is a dict with human-readable demographic descriptions as keys and  lists of census codes as values

    # dataframe to put our results in
    frame = pd.DataFrame(columns=request_codes.values(), index=list(demo_codes.keys()))

    # make a calculator object to do the calculations
    c = calculator()

    # populate the dataframe
    for request_code in list(request_codes.keys()):
        for demographic in list(demo_codes.keys()):
            avg_time_in_sec = c.get_avg_across_demographics(demo_codes[demographic],request_code)

            if avg_time_in_sec is None:
                #this can happen if, among the block groups that have put out 311 calls of type request_code, no members of the demographic live in that block group.
                # for example, no indigenous non-hispanic people live in any block group that has put out an "inacurate retail scales" 311 request.
                frame.loc[demographic, request_codes[request_code]] = 'N/A'
            else:
                delta = datetime.timedelta(seconds=avg_time_in_sec)
                frame.loc[demographic, request_codes[request_code]] = str(delta)

    # save the dataframe
    frame.to_csv('avg_times_by_demo_and_type.csv')

produce_table()