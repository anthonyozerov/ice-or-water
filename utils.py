import numpy as np
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point


# get closest indices for a list of values in a list of coordinates
def get_idx(vals, coords):
    vals = np.array(vals)
    idxs = np.zeros(len(vals))
    for i, v in tqdm(enumerate(vals)):
        idxs[i] = np.argmin(np.abs(v - coords))
    return idxs


# get data from an xarray using a list of coordinates
def get_data(x, y, array):
    x_idx = get_idx(x, array["x"].data)
    y_idx = get_idx(y, array["y"].data)
    data = []
    mat = np.array(array)
    for ix, iy in tqdm(zip(x_idx, y_idx)):
        data.append(mat[int(iy), int(ix)])
    return np.array(data)

# get a np array from an xarray using a range of coordinates
def get_image(xrange, yrange, array):

    x_idx_min, x_idx_max = get_idx([min(xrange), max(xrange)], array["x"].data)
    y_idx_min, y_idx_max = get_idx([max(yrange), min(yrange)], array["y"].data)

    data = np.array(array)

    subset = data[int(y_idx_min) : int(y_idx_max), int(x_idx_min) : int(x_idx_max)]

    extent = (min(xrange), max(xrange), min(yrange), max(yrange))

    return np.array(subset), extent

def get_points_within(df, polygon):

    point_df = gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(df["x"], df["y"])])
    polygon_df = gpd.GeoDataFrame(geometry=[polygon])
    join_left_df = point_df.sjoin(polygon_df, how="left")
    within = join_left_df["index_right"] == 0

    return within

