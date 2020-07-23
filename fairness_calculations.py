import numpy as np
import pandas as pd
import csv
import datetime
import math
import matplotlib.pyplot as plt
import matplotlib.figure as fig
from textwrap import wrap
from scipy.stats import gaussian_kde

count_syntax = ' - count'

class calculator:

    def __init__(self, meta_demographics_path='broad_demographic_categories_small.csv'):
        self.key_metric_table = pd.DataFrame
        # key_metric_table should be a table with block groups as rows and request types as columns,
        # where index (i,j) is block group i's measure of the metric for requests of type j
        # an example of a "key metric" might be average wait time, for example

        self.demographics = pd.read_csv('data preparation/demographics_table.csv', index_col=0)
        self.demographics.index = self.demographics.index.astype(np.str)
        self.demographics.fillna(0, inplace=True)

        self.request_counts = pd.read_csv('data preparation/request_count_table.csv', index_col=0)
        self.request_counts.fillna(0, inplace=True)

        self.geoids = [str(x) for x in self.request_counts.index]
        self.request_types = self.request_counts.columns

        self.meta_demographics = pd.DataFrame()
        self.meta_demographic_labels = list
        self.load_meta_demographics(meta_demographics_path)

    def load_metric_table(self, path):
        self.key_metric_table = pd.read_csv(path, index_col=0)
        self.key_metric_table.fillna(0, inplace=True)

    def load_meta_demographics(self, path):
        with open(path) as f:
            reader = csv.reader(f)
            demo_codes = {row[0]: row[1:] for row in reader}
            self.meta_demographic_labels = demo_codes.keys()
            demo_codes['Total'] = ['B03002001']
        self.meta_demographics = pd.DataFrame(index=self.geoids, columns=demo_codes.keys())

        for d in demo_codes.keys():
            temp = self.demographics.loc[:,[c + count_syntax for c in demo_codes[d]]]
            temp['sum'] = temp.sum(axis=1)
            self.meta_demographics[d] = temp['sum']

        for d in self.meta_demographic_labels:
            self.meta_demographics[d + ' ratio'] = self.meta_demographics.apply(lambda row: 0 if row['Total'] == 0 else row[d] / row['Total'], axis=1)

    def make_table(self, output_path, castastime=False):

        table = pd.DataFrame(index=self.meta_demographic_labels, columns=self.request_types)
        for d in self.meta_demographic_labels:
            for r in self.request_types:

                denominator = (self.request_counts[r] * self.meta_demographics[d + ' ratio']).sum()
                if denominator == 0:
                    table.loc[d, r] = 'N/A'
                    continue

                numerator = (self.key_metric_table[r] * self.request_counts[r] * self.meta_demographics[d + ' ratio']).sum()

                if castastime:
                    table.loc[d,r] = str(datetime.timedelta(seconds=numerator / denominator))
                else:
                    table.loc[d, r] = numerator / denominator

        table.to_csv(output_path, index=True)


    def estimate_distributions(self, col, scale, title_prefix, x_axis_label,write_path_prefix):

        # get a dictionary mapping short request codes to their human-readable names
        code_dict = pd.read_csv('data preparation/service_request_short_codes.csv', index_col=0).to_dict()['SR_TYPE']

        # get a list of demographics
        demos = self.meta_demographic_labels - ['Total']

        # make a graph showing PDFs by demographic for all request types
        for code in self.request_types:
            print(code)
            # read in the data for this request code
            frame = pd.read_csv('data preparation/311 data by request type/311_' + code + ".csv",
                                dtype={'CITY': np.str, 'STATE': np.str, 'ZIP_CODE': np.str,
                                       'STREET_NUMBER': np.str, 'LEGACY_SR_NUMBER': np.str,
                                       'PARENT_SR_NUMBER': np.str, 'SANITATION_DIVISION_DAYS': np.str,
                                       'BLOCK_GROUP': np.str},
                                index_col=0)

            # there are ~20 points across all the data that are right on the border with indiana,
            # and technically aren't in a block group. somehow at this point the block group for these
            # guys is just "n". So filter those out.
            frame = frame[frame['BLOCK_GROUP'] != 'n']

            # scale the points by a constant
            frame['scaled_vals'] = frame[col] / scale

            # figure out the scale the graph should have
            min_x = frame['scaled_vals'].min()
            max_x = frame['scaled_vals'].max()

            # get the matplotlib objects needed to make the graph
            fig, ax = plt.subplots(figsize=(6, 6.5))

            # make a different PDF for each demographic
            for d in demos:

                # get weight each point via per-block group demographic data
                weights = frame.apply(lambda row: self.meta_demographics.loc[row['BLOCK_GROUP'], d + ' ratio'], axis=1)

                # figurue out how many "people" we have
                effective_n = weights.sum().round(2)

                # only make a distribution if there's enough data to do so
                if effective_n > 1:
                    # do a density estimate to get a kernel
                    kernel = gaussian_kde(frame['scaled_vals'], weights=weights)
                    # make points to evaluate the kernel on
                    ind = np.linspace(min_x, max_x, 200)
                    # create the label for the kernel
                    label = f'{d} (n = {effective_n})'
                    # plot the kernel as a probability density function
                    ax.plot(ind, kernel.evaluate(ind), label=label)

            # formatting stuff
            ax.legend()
            title = f'{title_prefix} for {code_dict[code]}  requests (code {code})'
            plt.title('\n'.join(wrap(title, 60)))
            plt.xlabel(x_axis_label)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.80])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True)

            # save as an image
            plt.savefig(f'{write_path_prefix}{code}.png')
            plt.close('all')


    def estimate_distributions_noweights(self, threshold_n, x_axis_label, scale, write_path_prefix, title_prefix):

        # grab a mapping from request codes to human-readable names (used for labeling the distributions)
        code_dict = pd.read_csv('data preparation/service_request_short_codes.csv', index_col=0).to_dict()['SR_TYPE']

        for code in self.request_types:
            demos = self.meta_demographic_labels - ['Total']
            points = {d: [] for d in demos}
            for d in demos:
                for b in self.geoids:

                    numpoints = round(self.request_counts.loc[b, code] * self.meta_demographics.loc[b, d + ' ratio'])
                    avg = self.key_metric_table.loc[b, code]
                    avg_scaled = avg / scale # 24 * 60 * 60 for avg in days

                    points[d] += [avg_scaled] * numpoints

            # make all lists the same length by padding with NaN
            # (otherwise we can't put them in a datafame)
            # also, do some trickery to update the lables to reflect the number of points
            max_len = max([len(points[d]) for d in demos])
            points2 = {}
            for d in demos:

                if len(points[d]) < threshold_n:
                    continue
                newname = d + ' ("n" = ' + str(len(points[d])) + ')'
                points2[newname] = points[d] + ([np.nan] * (max_len - len(points[d])))

            # for tiny datasets, it may be that no demographics have enough data points to pass the threshhold
            # if that's the case, we can't calculate any distributions, so continue
            if len(points2.keys()) == 0:
                continue


            t =  f'{title_prefix} for {code_dict[code]}  requests (code {code})'

            # otherwise, put the points into a dataframe and PLOT that mf
            df = pd.DataFrame(points2)
            try:
                df.plot.kde(title=t, figsize=[6.4, 7])
            except np.linalg.LinAlgError:
                continue
            plt.xlabel(x_axis_label)

            plt.title('\n'.join(wrap(t,60)))
            ax = plt.subplot(111)

            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.80])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True)

            plt.savefig(write_path_prefix + code + '.png')
            plt.close('all')

    #
    # def make_histograms(self):
    #
    #     # get the request type codes
    #     request_types = pd.read_csv('data preparation/service_request_short_codes.csv')
    #     sr_codes = [x[0] for x in request_types[['SR_SHORT_CODE']].values]
    #
    #     for code in sr_codes:
    #         print(code)
    #         if code == 'SKA': continue
    #         request_data = pd.read_csv('data preparation/311 data by request type/311_' + code + ".csv",
    #                                    dtype={'CITY': np.str, 'STATE': np.str, 'ZIP_CODE': np.str,
    #                                           'STREET_NUMBER': np.str, 'LEGACY_SR_NUMBER': np.str,
    #                                           'PARENT_SR_NUMBER': np.str, 'SANITATION_DIVISION_DAYS': np.str},
    #                                    index_col=0)
    #
    #         max_time_sec = request_data['DELTA'].max()
    #         max_time_min_r10 = round(max_time_sec / (60 * 10)) * 10
    #
    #         new_frame = pd.DataFrame(data = 0, index=range(0, max_time_min_r10 + 10, 10), columns=self.meta_demographic_labels)
    #
    #
    #         request_data['delta in min r10'] = round(request_data['DELTA'] / (60 * 10)) * 10
    #
    #
    #         vcs = request_data.groupby('BLOCK_GROUP')['delta in min r10'].value_counts()
    #
    #         for bg in self.geoids:
    #             if bg in vcs.index:
    #                 for d in self.meta_demographic_labels:
    #
    #                     new_frame['temp'] = vcs[bg].squeeze()
    #                     new_frame['temp'] = new_frame.temp.fillna(0)
    #                     new_frame[d] += new_frame['temp'] * self.meta_demographics[d + ' ratio']
    #                     new_frame.drop(columns=['temp'], inplace=True)
    #
    #         new_frame.replace(to_replace=0, value=np.nan, inplace=True)
    #         new_frame.plot()
    #         plt.show()

    def get_avg_by_demo_and_request_type_old(self, demographic, request_type):

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


    def get_avg_across_demographics_old(self, demo_list, request_type):
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
            avg_time, raw_pop_sum = self.get_avg_by_demo_and_request_type_old(demographic, request_type)

            if avg_time is not None:
                sum += avg_time * raw_pop_sum
                total_pop_sum += raw_pop_sum

        if total_pop_sum == 0: return None
        return sum / total_pop_sum


# the below commented-out code is what I used to make the tables of average cue cuts per demographic/ request type.


input_path_prefix = 'data preparation/average x by block group/queue_displacement_'
output_path_prefix = 'avg_queue_displacement_'
bucket_types = ['3HR', '6HR', '24HR']

a1 = ['data preparation/average x by block group/bg_averages_raw_time.csv']
a2 = ['avg_raw_wait_time.csv']

path_info = zip([input_path_prefix + x + '.csv' for x in bucket_types], [output_path_prefix + y + '.csv' for y in bucket_types])


c = calculator()
c.estimate_distributions('DELTA', 60 * 60 * 24,'Wait time distribution ', 'Time (days)','wait time distributions/311_')


# c.load_metric_table(a1[0])
#c.estimate_distributions(4, 'wait time (days)',24 * 60 * 60, 'wait time distributions -- fuzzy method/wait_time_dist_', 'Wait time distribution ')

# for x in bucket_types:
#     c.load_metric_table(input_path_prefix + x + '.csv')
#     c.estimate_distributions(4, 'displacement in queue (positive = pushed back)', 1, 'queue displacement distributions/' + x + ' buckets/displacement_dist_' + x + '_', 'displacement distribution (' + x + ' buckets)')
#for l in bucket_types:

# for x in path_info:
#     print(x[0], x[1])
#
#     c.load_metric_table(x[0])
#
#     c.make_table(x[1])

# c.load_metric_table(a1[0])
# c.make_table(a2[0], castastime=True)

