import pandas as pd
import gzip
import json
import zipfile

import geopandas
from shapely.geometry import Point, Polygon
from planner.utils.files import resources


def load_schools(filename=resources / 'schools.zip'):
    with zipfile.ZipFile(filename) as zf:
        with zf.open('schools.json') as sch_stream:

            schools_data = json.load(sch_stream)

            schools_df_list = list()
            latitude = list()
            longitude = list()

            for r in schools_data:
                copy_fields = [
                    'OrgType',
                    'EducationPrograms',
                    'LegalOrganization',
                    'ReorganizationStatus',
                    'IDEKIS',
                    'InstitutionsAddresses',
                    'global_id',
                ]
                row_data = {
                    k: r.get(k) for k in copy_fields
                }
                row_data['data-type'] = 'school'

                students = r.get('NumberofStudentsInOO')
                quantity = None
                if students:
                    quantity = row_data['PupilsQuantity'] = students[0]['PupilsQuantity']

                long, lat = r['geodata_center']['coordinates']

                if quantity:
                    schools_df_list.append(row_data)
                    latitude.append(lat)
                    longitude.append(long)

    prepared_schools_df = geopandas.GeoDataFrame(schools_df_list,
                                                 geometry=geopandas.points_from_xy(longitude, latitude))
    return prepared_schools_df


def load_point_geometry(coords):
    if isinstance(coords[0], list):
        coords_data = list()
        for lat_long_row in coords:
            if len(lat_long_row) > 1:
                lat, long = lat_long_row
                coords_data.append([float(long), float(lat)])
        return Polygon(coords_data)
    else:
        #         print(coords)
        lat_long_row = coords
        if len(lat_long_row) > 1:
            lat, long = lat_long_row
            long, lat = float(long), float(lat)

            return Point(long, lat)


def load_eco_data(filename=resources / 'moskow-eco.json.gz'):
    with gzip.open(filename) as sch_stream:
        ecodata = json.load(sch_stream)

    res = pd.DataFrame(ecodata)

    del res['Color']

    del res['Image']

    res.rename(columns={'Type': 'data_type'}, inplace=True)

    geometry = res.Coords.apply(load_point_geometry)
    res = geopandas.GeoDataFrame(res, geometry=geometry)
    return res


def load_buildings_data(filename=resources / 'moskow-buildings.json.gz'):
    with gzip.open(filename) as sch_stream:
        ecodata = json.load(sch_stream)

    res = pd.DataFrame(ecodata['Objects'])

    res.rename(columns={'Type': 'data_type'}, inplace=True)

    geometry = geopandas.points_from_xy(res.Lng, res.Lat)
    res['data_type'] = 'Building'
    res = geopandas.GeoDataFrame(res, geometry=geometry)

    return res


def load_highways(filename=resources / 'highways.gz.pq'):
    return geopandas.read_parquet(filename)


def load_water(filename=resources / 'waters.gz.pq'):
    return geopandas.read_parquet(filename)
