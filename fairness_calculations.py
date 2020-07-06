import numpy as np
import pandas as pd
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


        r = self.response_times[request_type]

        d = self.demographics[demographic + count_syntax]

        combination = pd.concat([r,d], axis=1)

        combination = combination.dropna(axis=0, subset=['GRAF'])

        bg_populations = combination[demographic + count_syntax]
        bg_responsetimes = combination[request_type]

        if returnmode == 'avg':
            return(bg_populations.dot(bg_responsetimes) / bg_populations.sum())
        elif returnmode == 'detail':
            return(bg_populations.dot(bg_responsetimes), bg_populations.sum())


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
            dot_prod, pop_sum = self.get_avg_by_demo_and_request_type(demographic, request_type, returnmode='detail')
            sum += dot_prod * pop_sum
            total_pop_sum += pop_sum

        return sum / total_pop_sum


x = calculator()
print(x.get_avg_by_demo_and_request_type('B03002002','GRAF'))

print(x.get_avg_by_demo_and_request_type('B03002012','GRAF'))
