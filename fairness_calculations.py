import numpy as np
import pandas as pd
count_syntax = ' - count'

class calculator:

    def __init__(self):
        self.response_times = pd.read_csv('bg_averages.csv', index_col=0)
        self.demographics = pd.read_csv('demographics_table.csv', index_col=0)



    def get_avg_by_demo_and_request_type(self, demographic, request_type):

        ''' this function takes in a demographic a request type, and returns the average 311 response time'''

        r = self.response_times[request_type]

        d = self.demographics[demographic + count_syntax]

        combination = pd.concat([r,d], axis=1)

        combination = combination.dropna(axis=0, subset=['GRAF'])

        bg_populations = combination[demographic + count_syntax]
        bg_responsetimes = combination[request_type]

        return(bg_populations.dot(bg_responsetimes) / bg_populations.sum())


        # I used the below code to double-check that I was using pandas to calculate the dot product correctly. Keeping it for posterity.

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

x = calculator()
print(x.get_avg_by_demo_and_request_type('B03002002','GRAF'))

print(x.get_avg_by_demo_and_request_type('B03002012','GRAF'))
