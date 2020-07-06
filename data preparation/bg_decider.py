import shapefile
from shapely.geometry import Point, Polygon
from collections import namedtuple

cook_county_fp = '031'
chicago_coords = (-87.629, 41.8781)


'''
Note: this code expects the most recent TIGER/Line shapefiles for Illinois 
(downloadable at https://www.census.gov/cgi-bin/geo/shapefiles/index.php)
to be unzipped in a directory called "shapefiles".
'''

class bg_decider:

    def __init__(self):

        # load shapefile
        self.sf = shapefile.Reader("shapefiles/tl_2019_17_bg")
        self.bg_data = self.sf.shapeRecords()

        # as a pre-processing step, make polygon objects for each block group
        # and also filter out block groups that aren't in cook county

        bg_with_polygon = namedtuple('bg_with_polygon',['bg', 'polygon'])

        self.bgs_with_polygons = []
        for x in self.bg_data:
            if x.record.COUNTYFP == cook_county_fp:
                self.bgs_with_polygons.append(bg_with_polygon(x, Polygon(x.shape.points)))


    def decide_bg(self, latitude, longitude, returnmode='geoid'):

        if returnmode not in ['geoid', 'object', 'bg id']:
            raise NotImplementedError('return mode ' + returnmode + ' is not valid.')

        point = Point(longitude, latitude)
        for bg_w_poly in self.bgs_with_polygons:
            if point.within(bg_w_poly.polygon):


                if returnmode == 'geoid':
                    return bg_w_poly.bg.record.GEOID
                if returnmode == 'object':
                    return bg_w_poly.bg
                if returnmode == 'bg id':
                    return bg_w_poly.bg.record.BLKGRPCE

        print(latitude, longitude, 'outside of cook county')
        return None
        # raise ValueError(str(latitude) + " " + str(longitude) + " not found within a block group in illinois")