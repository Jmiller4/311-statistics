import numpy as np
import pandas as pd
import csv
import datetime as dt
import math
import matplotlib.pyplot as plt
import matplotlib.figure as fig
from textwrap import wrap
from scipy.stats import gaussian_kde
import random
import json

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

        self.meta_demographics_table = pd.DataFrame()
        self.demographic_labels = []
        self.demo_codes = {}
        self.line_colors = {}
        self.load_meta_demographics(meta_demographics_path)

    def load_metric_table(self, path):
        self.key_metric_table = pd.read_csv(path, index_col=0)
        self.key_metric_table.fillna(0, inplace=True)

    def load_meta_demographics(self, path):
        with open(path) as f:
            reader = csv.reader(f)
            self.demo_codes = {row[0]: row[1:] for row in reader}
            self.demographic_labels = list(self.demo_codes.keys())
        self.meta_demographics_table = pd.DataFrame(index=self.geoids, columns=self.demo_codes.keys())

        for d in self.demo_codes.keys():
            temp = self.demographics.loc[:,[c + count_syntax for c in self.demo_codes[d]]]
            temp['sum'] = temp.sum(axis=1)
            self.meta_demographics_table[d] = temp['sum']

        self.meta_demographics_table['Total'] = self.demographics['B03002001' + count_syntax]

        for d in self.demographic_labels:
            self.meta_demographics_table[d + ' ratio'] = self.meta_demographics_table.apply(lambda row: 0 if row['Total'] == 0 else row[d] / row['Total'], axis=1)

        # decide which color line will represent each demographic.
        # doing this makes it easier to look at multiple distributions --
        # it saves one the trouble of re-reading the legend each time
        col_list = ['c','y','g','r','m','k','c']
        self.demographic_labels.sort()
        for x in self.demographic_labels:
            if len(col_list) > 0:
                self.line_colors[x] = col_list.pop(0)
            else:
                # if we assign a demographic "None" as a color,
                # that really means we're going to let matplotlib
                # automatically assign a color
                self.line_colors[x] = None

    def make_table(self, output_path, castastime=False):

        table = pd.DataFrame(index=self.demographic_labels, columns=self.request_types)
        for d in self.demographic_labels:
            for r in self.request_types:

                denominator = (self.request_counts[r] * self.meta_demographics_table[d + ' ratio']).sum()
                if denominator == 0:
                    table.loc[d, r] = 'N/A'
                    continue

                numerator = (self.key_metric_table[r] * self.request_counts[r] * self.meta_demographics_table[d + ' ratio']).sum()

                if castastime:
                    table.loc[d,r] = str(dt.timedelta(seconds=numerator / denominator))
                else:
                    table.loc[d, r] = numerator / denominator

        table.to_csv(output_path, index=True)


    def estimate_distribution(self, code, col, scale, title_prefix, x_axis_label,write_path_prefix, graph_scale_percentile=100):

        # get a dictionary mapping short request codes to their human-readable names
        code_dict = pd.read_csv('data preparation/service_request_short_codes.csv', index_col=0).to_dict()['SR_TYPE']

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
        # this is useful in the case where the column we're focusing on for the distribution is a time in seconds,
        # but we want to make a distribution over time in days.
        frame['scaled_vals'] = frame[col] / scale

        # figure out the scale the graph should have
        min_x = frame['scaled_vals'].min()
        max_x = frame['scaled_vals'].quantile(q=graph_scale_percentile / 100)

        # get the matplotlib objects needed to make the graph
        fig, ax = plt.subplots(figsize=(6, 6.5))

        # make a different PDF for each demographic
        for d in self.demographic_labels:

            # get weight each point via per-block group demographic data
            weights = frame.apply(lambda row: self.meta_demographics_table.loc[row['BLOCK_GROUP'], d + ' ratio'], axis=1)

            # figurue out how many "people" we have
            effective_n = weights.sum().round(4)

            # only make a distribution if there's enough data to do so
            if effective_n > 1:
                # do a density estimate to get a kernel
                kernel = gaussian_kde(frame['scaled_vals'], weights=weights)
                # make points to evaluate the kernel on
                ind = np.linspace(min_x, max_x, 200)
                # create the label for the kernel
                label = f'{d} (n = {effective_n})'
                # plot the kernel as a probability density function,
                # assigning the line a color if possible
                if self.line_colors[d] is not None:
                    ax.plot(ind, kernel.evaluate(ind), label=label, color=self.line_colors[d])
                else:
                    ax.plot(ind, kernel.evaluate(ind), label=label)

        # formatting stuff
        ax.legend()
        title = f'{title_prefix} for {code_dict[code]}  requests (code {code})'
        if graph_scale_percentile != 100:
            title += f', graph covers {graph_scale_percentile}th percentile'
        plt.title('\n'.join(wrap(title, 60)))
        plt.xlabel(x_axis_label)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.80])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True)

        # make the filename and save as an image

        # first, take the slashes out of the code description
        code_description = code_dict[code].replace('/', '(slash)')
        fname = f'{write_path_prefix}{code_description}'
        if graph_scale_percentile != 100:
            fname += f'_{graph_scale_percentile}'
        fname += '.png'
        plt.savefig(fname)
        plt.close('all')


    def estimate_distributions_for_all_requests(self, col, scale, title_prefix, x_axis_label,write_path_prefix, graph_scale_percentile=100):

        # make a graph showing PDFs by demographic for all request types
        for code in self.request_types:
            print(code)
            self.estimate_distribution(code, col, scale, title_prefix, x_axis_label, write_path_prefix, graph_scale_percentile)


    def estimate_distributions_noweights(self, threshold_n, x_axis_label, scale, write_path_prefix, title_prefix):

        # grab a mapping from request codes to human-readable names (used for labeling the distributions)
        code_dict = pd.read_csv('data preparation/service_request_short_codes.csv', index_col=0).to_dict()['SR_TYPE']

        for code in self.request_types:

            points = {d: [] for d in self.demographic_labels}
            for d in self.demographic_labels:
                for b in self.geoids:

                    numpoints = round(self.request_counts.loc[b, code] * self.meta_demographics_table.loc[b, d + ' ratio'])
                    avg = self.key_metric_table.loc[b, code]
                    avg_scaled = avg / scale # 24 * 60 * 60 for avg in days

                    points[d] += [avg_scaled] * numpoints

            # make all lists the same length by padding with NaN
            # (otherwise we can't put them in a datafame)
            # also, do some trickery to update the lables to reflect the number of points
            max_len = max([len(points[d]) for d in self.demographic_labels])
            points2 = {}
            for d in self.demographic_labels:

                if len(points[d]) < threshold_n:
                    continue
                newname = f'{d} ("n" = {len(points[d])})'

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



input_path_prefix = 'data preparation/average x by block group/queue_displacement_'
output_path_prefix = 'avg_queue_displacement_'
bucket_types = ['3HR', '6HR', '24HR']

a1 = ['data preparation/average x by block group/bg_averages_raw_time.csv']
a2 = ['avg_raw_wait_time.csv']

path_info = zip([input_path_prefix + x + '.csv' for x in bucket_types], [output_path_prefix + y + '.csv' for y in bucket_types])


c = calculator()
c.estimate_distributions_for_all_requests('DELTA', 60 * 60 * 24,'Wait time distribution ', 'Time (days)','wait time distributions/')
c.estimate_distributions_for_all_requests('DELTA', 60 * 60 * 24,'Wait time distribution ', 'Time (days)','wait time distributions/', graph_scale_percentile=95)

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

