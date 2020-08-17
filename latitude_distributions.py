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

cermak = 41.852801
ninetyfifth_st = 41.721782
grand = 41.891051
fullerton = 41.925412 #depaul
touhy_ave = 42.012514 #rogers park

def latitude_weight(lat):

    min_lat = 41.644567935
    max_lat = 42.022945905

    range = max_lat - min_lat
    normalized_lat = (lat - min_lat) / range
    return normalized_lat


def draw_latitude_distributions():

    helper_lines = [[41.721782, '95th St.','k'],
                    [41.785942, '60th St. (U Chicago)', 'g'],
                    [41.852801, 'Cermak','y'],
                    #[41.891051, 'Grand Ave','g'],
                    [41.925412, 'Fullerton Ave (DePaul)','r'],
                    [42.012514, 'Touhy ave (Rogers Park)','c']]


    code_dict = pd.read_csv('data preparation/service_request_short_codes.csv', index_col=0).to_dict()['SR_TYPE']

    codes = code_dict.keys()

    for code in codes:

        frame = pd.read_csv('data preparation/311 data by request type/311_' + code + ".csv",
                            dtype={'CITY': np.str, 'STATE': np.str, 'ZIP_CODE': np.str,
                                   'STREET_NUMBER': np.str, 'LEGACY_SR_NUMBER': np.str,
                                   'PARENT_SR_NUMBER': np.str, 'SANITATION_DIVISION_DAYS': np.str,
                                   'BLOCK_GROUP': np.str},
                            index_col=0)




        frame['normalized latitudes'] = frame.apply(lambda row: latitude_weight(row['LATITUDE']), axis=1)

        # do a density estimate to get a kernel
        try:
            kernel = gaussian_kde(frame['normalized latitudes'])
        except np.linalg.LinAlgError:
            continue

        # make points to evaluate the kernel on
        ind = np.linspace(0, 1, 200)


        for gtype in ['PDF', 'CDF', 'Quantile Plot']:

            # get the matplotlib objects needed to make the graph
            fig, ax = plt.subplots(figsize=(6, 6.5))

            if gtype == 'PDF':
                ax.plot(ind, kernel.evaluate(ind), label=gtype, color='b')
            else:
                cdf_points = [kernel.integrate_box_1d(np.NINF, x) for x in ind]
                if gtype == 'CDF':
                    ax.plot(ind, cdf_points, label=gtype, color='b')
                else:
                    ax.plot(cdf_points, ind, label=gtype, color='b')

            # draw in "helper" guide lines

            fn = plt.axvline
            if gtype == 'Quantile Plot': fn = plt.axhline
            for x in helper_lines:
                fn(latitude_weight(x[0]), label=x[1], color=x[2], linestyle='--')

            # formatting stuff
            ax.legend()
            title = f'Latitude {gtype} for {code_dict[code]}  requests (code {code})'
            plt.title('\n'.join(wrap(title, 60)))


            if gtype != 'Quantile Plot':
                plt.xlabel('Normalized latitude (0 = farthest south, 1 = farthest north)')
            else:
                plt.xlabel('Quantile')
                plt.ylabel('Normalized latitude (0 = farthest south, 1 = farthest north)')
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.80])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2)

            # name the file and save as an image

            # first, take the slashes out of the code description
            code_description = code_dict[code].replace('/', '(slash)')
            fname = f'Request amount distributions/{gtype}s/{code_description}.png'
            plt.savefig(fname)
            plt.close('all')

draw_latitude_distributions()

