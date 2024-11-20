import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Polygon
import os
import xarray as xr
import numpy as np

def load_ground(xrange, yrange):
    basins = gpd.read_file("data/basins/ANT_Basins_IMBIE2_v1.6")
    grounding_line = unary_union(basins.geometry)

    min_x = xrange[0] - 1e3
    max_x = xrange[1] + 1e3
    min_y = yrange[0] - 1e3
    max_y = yrange[1] + 1e3

    bbox = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])

    grounding_line = grounding_line.intersection(bbox)

    return grounding_line

def load_earthaccess(short_name, version):
    import earthaccess

    earthaccess.login(strategy="interactive", persist=True)
    fs = earthaccess.get_fsspec_https_session()
    granules = earthaccess.search_data(short_name=short_name, version=version)
    metadata = earthaccess.consolidate_metadata(
        granules, access="indirect", kerchunk_options={"coo_map": []}
    )
    data = xr.open_dataset(
        "reference://",
        engine="zarr",
        chunks="auto",
        backend_kwargs={
            "consolidated": False,
            "storage_options": {
                "fo": metadata,
                "remote_protocol": "https",
                "remote_options": fs.storage_options,
            },
        },
    )
    return data


def load_icevelocity(xrange, yrange):
    if os.path.exists("data/antarctica_ice_velocity_450m_v2.nc"):
        vel = xr.open_dataset("data/antarctica_ice_velocity_450m_v2.nc", chunks="auto")
    else:
        vel = load_earthaccess(short_name="NSIDC-0484", version="2")

    min_x = xrange[0]
    max_x = xrange[1]
    min_y = yrange[0]
    max_y = yrange[1]

    vel_clip = vel.sel(x=slice(min_x, max_x), y=slice(max_y, min_y))
    vel_mag = np.sqrt(vel_clip["VX"] ** 2 + vel_clip["VY"] ** 2)
    vel_mag = vel_mag.compute()
    vel = vel_mag
    return vel

def load_bed(xrange, yrange):
    if os.path.exists("data/BedMachineAntarctica-v3.nc"):
        bed = xr.open_dataset("data/BedMachineAntarctica-v3.nc", chunks="auto")
    else:
        bed = load_earthaccess(short_name="NSIDC-0756", version="3")

    min_x = xrange[0]
    max_x = xrange[1]
    min_y = yrange[0]
    max_y = yrange[1]

    bed_clip = bed.sel(x=slice(min_x, max_x), y=slice(max_y, min_y))
    bed = bed_clip.compute()

    bedalt = bed["bed"]
    thick = bed["thickness"]
    return bedalt, thick
