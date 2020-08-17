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

        self.kernels = {}

    def load_metric_table(self, path):
        self.key_metric_table = pd.read_csv(path, index_col=0)
        self.key_metric_table.fillna(0, inplace=True)

    def load_meta_demographics(self, path):
        with open(path) as f:
            reader = csv.reader(f)
            self.demo_codes = {row[0]: row[1:] for row in reader}
            self.demographic_labels = list(self.demo_codes.keys())

        # meta_demographics_table is a dataframe with block groups as rows and "meta-demographics" (groups of disjoint ACS demographics) as columns
        # specifically, for each block group and each meta-demographic, we want to get the FRACTION of people in that block group
        #

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
        col_list = ['c','r','y','g','m','k']
        self.demographic_labels.sort()
        for x in self.demographic_labels:
            if len(col_list) > 0:
                self.line_colors[x] = col_list.pop(0)
            else:
                # if we assign a demographic "None" as a color,
                # that really means we're going to let matplotlib
                # automatically assign a color later
                self.line_colors[x] = None

    def make_averages_table(self, output_path, castastime=False):

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


    def load_kernels(self,col, categories, weight_fn, weight_fn_arg, scale=1):

        for code in self.request_types:

            self.kernels[code] = {}

            frame = pd.read_csv('data preparation/311 data by request type/311_' + code + ".csv",
                                dtype={'CITY': np.str, 'STATE': np.str, 'ZIP_CODE': np.str,
                                       'STREET_NUMBER': np.str, 'LEGACY_SR_NUMBER': np.str,
                                       'PARENT_SR_NUMBER': np.str, 'SANITATION_DIVISION_DAYS': np.str,
                                       'BLOCK_GROUP': np.str},
                                index_col=0)
            if weight_fn == self.assign_weights_based_on_bg_demos:
                frame = frame[frame['BLOCK_GROUP'] != 'n']
            frame['scaled_vals'] = frame[col] / scale

            for category in categories:

                self.kernels[code][category] = {}

                # get weight each point via per-block group demographic data
                weights = frame.apply(lambda row: weight_fn(category, row, weight_fn_arg), axis=1)

                # figurue out how many "people" we have
                effective_n = weights.sum().round(4)

                self.kernels[code][category]['n'] = effective_n

                # don't make a kernel if there's not enough "weight"/ data
                if effective_n <= 1:
                    self.kernels[code][category]['k'] = 'N/A'
                    continue

                # do a density estimate to get a kernel
                try:
                    kernel = gaussian_kde(frame['scaled_vals'], weights=weights)
                    self.kernels[code][category]['k'] = kernel
                except np.linalg.LinAlgError:
                    print(f'singular matrix for data from column {col}, code {code}, category {category}')
                    self.kernels[code][category]['k'] = 'N/A'


    def draw_charts(self, col,
                                   title_prefix, x_axis_label, write_path_prefix,
                                   categories, colors,
                                   scale=1, lower_percentile=0, upper_percentile=100,
                                   graph_type='pdf',legend_cols=1,y_axis_label=None):

        code_dict = pd.read_csv('data preparation/service_request_short_codes.csv', index_col=0).to_dict()['SR_TYPE']

        for code in self.request_types:


            frame = pd.read_csv('data preparation/311 data by request type/311_' + code + ".csv",
                                dtype={'CITY': np.str, 'STATE': np.str, 'ZIP_CODE': np.str,
                                       'STREET_NUMBER': np.str, 'LEGACY_SR_NUMBER': np.str,
                                       'PARENT_SR_NUMBER': np.str, 'SANITATION_DIVISION_DAYS': np.str,
                                       'BLOCK_GROUP': np.str},
                                index_col=0)

            frame['scaled_vals'] = frame[col] / scale

            # figure out the scale the graph should have
            min_x = frame['scaled_vals'].quantile(q=lower_percentile / 100)
            max_x = frame['scaled_vals'].quantile(q=upper_percentile / 100)

            if min_x == max_x:
                print(f'for code {code}, column {col}, '
                      f'lower percentile {lower_percentile} and upper percentile {upper_percentile},'
                      f' data has same minimum and maximum value: {min_x} -- so cannot make a distribution')
                continue


            # get the matplotlib objects needed to make the graph
            fig, ax = plt.subplots(figsize=(6, 6.5))

            # draw a different line for each demographic
            for c in categories:

                if self.kernels[code][c]['k'] != 'N/A':

                    kernel = self.kernels[code][c]['k']
                    effective_n = self.kernels[code][c]['n']
                    # make points to evaluate the kernel on
                    ind = np.linspace(min_x, max_x, 200)

                    # create the label for the kernel
                    label = f'{c} (n = {effective_n})'

                    # plot the kernel as a probability density function, or as a cumulative distribution function
                    # assigning the line a color if possible

                    if graph_type == 'cdf' or graph_type == 'quantile plot':
                        # find the cdf (a quantile plot is essentially the inverse of a cdf)
                        # the method "integrate_box_1d" gives the area under the pdf between the first and second argument
                        # so the following line of code gives a set of points representing the cdf
                        cdf_points = [kernel.integrate_box_1d(np.NINF, x) for x in ind]
                        # (np.NINF is a constant representing negative infinity)


                        x_axis_points = ind
                        y_axis_points = cdf_points

                        # if we're plotting a quantile plot (i.e. the inverse of the cdf),
                        # flip which points are put on the x axis and which are put on the y axis
                        if graph_type == 'quantile plot':
                            temp = x_axis_points
                            x_axis_points = y_axis_points
                            y_axis_points = temp

                        # plot the cdf/ quantile plot with that demographic's color, if we were able to assign it one
                        if colors[c] is not None:
                            ax.plot(x_axis_points, y_axis_points, label=label,
                                    color=colors[c])
                        else:
                            ax.plot(x_axis_points, y_axis_points, label=label)

                    if graph_type == 'pdf':
                        #plot a pdf with that demographic's color, if we were able to assign it one
                        #kernel.evaluate(ind) evaluates the kernel on each point in ind and returns those results as a list
                        if colors[c] is not None:
                            ax.plot(ind, kernel.evaluate(ind), label=label,
                                    color=colors[c])
                        else:
                            ax.plot(ind, kernel.evaluate(ind), label=label)

            # formatting stuff
            ax.legend()
            title = f'{title_prefix} for {code_dict[code]}  requests (code {code})'
            if upper_percentile != 100 or lower_percentile != 0:
                if lower_percentile != 0 and upper_percentile !=0:
                    title += f', graph covers {lower_percentile}th to {upper_percentile}th quantile'
                elif upper_percentile != 0:
                    title += f', graph covers {upper_percentile}th quantile'
                else:
                    title += f', graph starts at {lower_percentile}th quantile'
            plt.title('\n'.join(wrap(title, 60)))
            plt.xlabel(x_axis_label)
            if y_axis_label is not None:
                plt.ylabel(y_axis_label)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.80])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=legend_cols)

            # name the file and save as an image

            # first, take the slashes out of the code description
            code_description = code_dict[code].replace('/', '(slash)')
            fname = f'{write_path_prefix}{code_description}'
            if upper_percentile != 100 or lower_percentile != 0:
                if lower_percentile != 0:
                    fname += f'_{lower_percentile}'
                fname += f'_{upper_percentile}'
            fname += '.png'
            plt.savefig(fname)
            plt.close('all')


    def estimate_distributions_new(self, col,
                                   title_prefix, x_axis_label, write_path_prefix,
                                   categories, colors, weight_fn, weight_fn_arg,
                                   scale=1, lower_percentile=0, upper_percentile=100,
                                   graph_type='pdf',legend_cols=1,y_axis_label=None):
        code_dict = pd.read_csv('data preparation/service_request_short_codes.csv', index_col=0).to_dict()['SR_TYPE']



        for code in self.request_types:


            frame = pd.read_csv('data preparation/311 data by request type/311_' + code + ".csv",
                                dtype={'CITY': np.str, 'STATE': np.str, 'ZIP_CODE': np.str,
                                       'STREET_NUMBER': np.str, 'LEGACY_SR_NUMBER': np.str,
                                       'PARENT_SR_NUMBER': np.str, 'SANITATION_DIVISION_DAYS': np.str,
                                       'BLOCK_GROUP': np.str},
                                index_col=0)

            if weight_fn == self.assign_weights_based_on_bg_demos:
                frame = frame[frame['BLOCK_GROUP'] != 'n']

            frame['scaled_vals'] = frame[col] / scale

            # figure out the scale the graph should have
            min_x = frame['scaled_vals'].quantile(q=lower_percentile / 100)
            max_x = frame['scaled_vals'].quantile(q=upper_percentile / 100)

            if min_x == max_x:
                print(f'for code {code}, column {col}, '
                      f'lower percentile {lower_percentile} and upper percentile {upper_percentile},'
                      f' data has same minimum and maximum value: {min_x} -- so cannot make a distribution')
                continue


            # get the matplotlib objects needed to make the graph
            fig, ax = plt.subplots(figsize=(6, 6.5))

            # make a different PDF/CDF for each demographic
            for c in categories:

                # get weight each point via per-block group demographic data
                weights = frame.apply(lambda row: weight_fn(c, row, weight_fn_arg), axis=1)

                # figurue out how many "people" we have
                effective_n = weights.sum().round(4)

                # only make a distribution if there's enough data to do so
                if effective_n > 1:

                    # do a density estimate to get a kernel
                    try:
                        kernel = gaussian_kde(frame['scaled_vals'], weights=weights)
                    except np.linalg.LinAlgError:
                        print(f'singular matrix for data from column {col}, code {code}, category {c}')
                        continue


                    # make points to evaluate the kernel on
                    ind = np.linspace(min_x, max_x, 200)

                    # create the label for the kernel
                    label = f'{c} (n = {effective_n})'

                    # plot the kernel as a probability density function, or as a cumulative distribution function
                    # assigning the line a color if possible

                    if graph_type == 'cdf' or graph_type == 'quantile plot':
                        # find the cdf (a quantile plot is essentially the inverse of a cdf)
                        # the method "integrate_box_1d" gives the area under the pdf between the first and second argument
                        # so the following line of code gives a set of points representing the cdf
                        cdf_points = [kernel.integrate_box_1d(np.NINF, x) for x in ind]
                        # (np.NINF is a constant representing negative infinity)

                        x_axis_points = ind
                        y_axis_points = cdf_points

                        # if we're plotting a quantile plot (i.e. the inverse of the cdf),
                        # flip which points are put on the x axis and which are put on the y axis
                        if graph_type == 'quantile plot':
                            temp = x_axis_points
                            x_axis_points = y_axis_points
                            y_axis_points = temp

                        # plot the cdf/ quantile plot with that demographic's color, if we were able to assign it one
                        if colors[c] is not None:
                            ax.plot(x_axis_points, y_axis_points, label=label,
                                    color=colors[c])
                        else:
                            ax.plot(x_axis_points, y_axis_points, label=label)

                    if graph_type == 'pdf':
                        #plot a pdf with that demographic's color, if we were able to assign it one
                        #kernel.evaluate(ind) evaluates the kernel on each point in ind and returns those results as a list
                        if colors[c] is not None:
                            ax.plot(ind, kernel.evaluate(ind), label=label,
                                    color=colors[c])
                        else:
                            ax.plot(ind, kernel.evaluate(ind), label=label)

            # formatting stuff
            ax.legend()
            title = f'{title_prefix} for {code_dict[code]}  requests (code {code})'
            if upper_percentile != 100 or lower_percentile != 0:
                if lower_percentile != 0 and upper_percentile !=0:
                    title += f', graph covers {lower_percentile}th to {upper_percentile}th percentile'
                elif upper_percentile != 0:
                    title += f', graph covers {upper_percentile}th percentile'
                else:
                    title += f', graph starts at {lower_percentile}th percentile'
            plt.title('\n'.join(wrap(title, 60)))
            plt.xlabel(x_axis_label)
            if y_axis_label is not None:
                plt.ylabel(y_axis_label)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.80])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=legend_cols)

            # name the file and save as an image

            # first, take the slashes out of the code description
            code_description = code_dict[code].replace('/', '(slash)')
            fname = f'{write_path_prefix}{code_description}'
            if upper_percentile != 100 or lower_percentile != 0:
                if lower_percentile != 0:
                    fname += f'_{lower_percentile}'
                fname += f'_{upper_percentile}'
            fname += '.png'
            plt.savefig(fname)
            plt.close('all')

    def assign_weights_based_on_bg_demos(self, demo, row, dummy_arg):
        return self.meta_demographics_table.loc[row['BLOCK_GROUP'], demo + ' ratio']

    def estimate_distribution(self, code, col,
                              title_prefix, x_axis_label, write_path_prefix,
                              scale=1, lower_percentile=0, upper_percentile=100,
                              cdf=False):

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
        min_x = frame['scaled_vals'].quantile(q=lower_percentile / 100)
        max_x = frame['scaled_vals'].quantile(q=upper_percentile / 100)

        if min_x == max_x:
            print(f'for code {code}, column {col}, '
                  f'lower percentile {lower_percentile} and upper percentile {upper_percentile},'
                  f' data has same minimum and maximum value: {min_x} -- so cannot make a distribution')
            return


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

                # plot the kernel as a probability density function, or as a cumulative distribution function
                # assigning the line a color if possible

                if cdf:
                    # plot a cdf.
                    # the method "integrate_box_1d" gives the area under the pdf between the first and second argument
                    # so the following line of code gives a set of points representing the cdf
                    cdf_points = [kernel.integrate_box_1d(np.NINF, x) for x in ind]
                    # (np.NINF is a constant representing negative infinity)


                    # plot the cdf with that demographic's color, if we were able to assign it one
                    if self.line_colors[d] is not None:
                        ax.plot(ind, cdf_points, label=label,
                                color=self.line_colors[d])
                    else:
                        ax.plot(ind, cdf_points, label=label)

                else:
                    #plot a pdf with that demographic's color, if we were able to assign it one
                    #kernel.evaluate(ind) evaluates the kernel on each point in ind and returns those results as a list
                    if self.line_colors[d] is not None:
                        ax.plot(ind, kernel.evaluate(ind), label=label,
                                color=self.line_colors[d])
                    else:
                        ax.plot(ind, kernel.evaluate(ind), label=label)


        # formatting stuff
        ax.legend()
        title = f'{title_prefix} for {code_dict[code]}  requests (code {code})'
        if upper_percentile != 100 or lower_percentile != 0:
            if lower_percentile != 0 and upper_percentile !=0:
                title += f', graph covers {lower_percentile}th to {upper_percentile}th percentile'
            elif upper_percentile != 0:
                title += f', graph covers {upper_percentile}th percentile'
            else:
                title += f', graph starts at {lower_percentile}th percentile'
        plt.title('\n'.join(wrap(title, 60)))
        plt.xlabel(x_axis_label)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.80])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True)

        # name the file and save as an image

        # first, take the slashes out of the code description
        code_description = code_dict[code].replace('/', '(slash)')
        fname = f'{write_path_prefix}{code_description}'
        if upper_percentile != 100 or lower_percentile != 0:
            if lower_percentile != 0:
                fname += f'_{lower_percentile}'
            fname += f'_{upper_percentile}'
        fname += '.png'
        plt.savefig(fname)
        plt.close('all')

    def estimate_distributions_for_all_requests(self, col, title_prefix, x_axis_label,write_path_prefix, scale=1,
                                                lower_percentile=0, upper_percentile=100,
                                                cdf=False):

        # make a graph showing PDFs by demographic for all request types
        for code in self.request_types:
            print(code)
            self.estimate_distribution(code, col, title_prefix, x_axis_label, write_path_prefix, scale=scale,
                                       upper_percentile=upper_percentile, lower_percentile=lower_percentile,
                                       cdf=cdf)

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
bucket_types = ['1HR', '6HR', '24HR'] #1SEC, 3HR

a1 = ['data preparation/average x by block group/bg_averages_raw_time.csv']
a2 = ['avg_raw_wait_time.csv']

path_info = zip([input_path_prefix + x + '.csv' for x in bucket_types], [output_path_prefix + y + '.csv' for y in bucket_types])


with open('sides.csv') as sides_f:
    s_reader = csv.reader(sides_f)
    side_areanum_map = {row[0]: [int(x) for x in row[1:]] for row in s_reader}
with open('community_areas.csv') as area_f:
    a_reader = csv.reader(area_f)
    number_area_map = {int(row[0]): row[1:] for row in a_reader}
with open('meta_sides.csv') as meta_f:
    m_reader = csv.reader(meta_f)
    metaside_side_map = {row[0]: row[1:] for row in m_reader}


def side_weights(label, row, side_areanum_map):
    areanum = row['COMMUNITY_AREA']
    areas = side_areanum_map[label]
    return int(areanum in areas)

def metaside_weights(label, row, double):
    side_areanum_map = double[0]
    metaside_side_map = double[1]
    areanum = row['COMMUNITY_AREA']
    areas = list(pd.core.common.flatten([side_areanum_map[side] for side in metaside_side_map[label]]))
    return int(areanum in areas)

col_list = ['c','r','y','g','m','k','c']
metasides = list(metaside_side_map.keys())
metaside_color_map = {}
for m in metasides:
    metaside_color_map[m] = col_list.pop(0)

sides = list(side_areanum_map.keys())
sides_color_map = {}
for s in sides:
    if len(col_list) > 0:
        sides_color_map[s] = col_list.pop(0)
    else:
        sides_color_map[s] = None

def latitude_weight(label, row, dummy_arg):
    min_lat = 41.644567935
    max_lat = 42.022945905

    range = max_lat - min_lat
    normalized_lat = (row['LATITUDE'] - min_lat) / range
    if label == 'North':
        return normalized_lat
    elif label == 'South':
        return 1 - normalized_lat
    raise NotImplementedError

#
# for x in [41.644567935, 42.022945905, 41.7, 42.01]:
#     print(f'{x}, North: {latitude_weight("North",x,0)}')
#     print(f'{x}, South: {latitude_weight("South",x,0)}')


c = calculator()

# c.estimate_distributions_new('DELTA',
#                              'Wait time PDF',
#                              'Time (days)',
#                              'queue displacement distributions/by geography/consolidated sides/PDFs/',
#                              metasides, metaside_color_map, metaside_weights, (side_areanum_map, metaside_side_map),
#                              upper_percentile=95,
#                              graph_type='pdf', legend_cols=2)


#########################~~~~~~~~~~~

c.load_kernels('DELTA',c.demographic_labels,c.assign_weights_based_on_bg_demos,0, scale=24*60*60)

print('done loading')

for P in [100, 95]:
    # c.draw_charts('DELTA',
    #               'Wait time CDF','Time (days)',
    #               'wait time distributions/by geography/all sides/CDFs/',
    #               sides, sides_color_map,upper_percentile=P, graph_type='cdf',
    #               legend_cols=2,
    #               scale=24 * 60 * 60,
    #               y_axis_label=None)
    #
    # c.draw_charts('DELTA',
    #               'Wait time PDF','Time (days)',
    #               'wait time distributions/by geography/all sides/PDFs/',
    #               sides, sides_color_map,upper_percentile=P, graph_type='pdf',
    #               legend_cols=2,
    #               scale=24 * 60 * 60,
    #               y_axis_label=None)

    c.draw_charts('DELTA',
                  'Wait time Quantile Plot','Quantile',
                  'wait time distributions/by race and ethnicity/Quantile Plots/',
                  c.demographic_labels, c.line_colors, upper_percentile=P, graph_type='quantile plot',
                  legend_cols=1,
                  scale=24 * 60 * 60,
                  y_axis_label='Time (days)')


# c.estimate_distributions_new('DELTA','Wait time quantile plot','quantile','Geography/consolidated sides/Quantile Plots/',
#                              metasides, metaside_color_map, metaside_weights,(side_areanum_map, metaside_side_map),
#                              scale= 24 * 60 * 60,
#                              upper_percentile=100,
#                              graph_type='quantile plot', legend_cols=2,
#                              y_axis_label='Time (days)')


# c.estimate_distributions_new('DELTA','Wait time quantile plot','quantile','Geography/latitude/Quantile Plots/',
#                              ['North','South'], {'North':'r','South':'b'},latitude_weight,0,
#                              scale= 24 * 60 * 60,
#                              upper_percentile=100,
#                              graph_type='quantile plot', legend_cols=1,
#                              y_axis_label='Time (days)')


# c.estimate_distributions_new('CR_BUCKET_1HR_DISPLACEMENT',
#                              'Queue displacement PDF',
#                              'Queue displacement',
#                              'queue displacement distributions/by geography/consolidated sides/PDFs/',
#                              metasides, metaside_color_map, metaside_weights, (side_areanum_map, metaside_side_map),
#                              upper_percentile=100,
#                              graph_type='pdf', legend_cols=2)
