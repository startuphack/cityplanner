"""
В этом файле собраны загрузчики различных данных, школы, железные дороги и т.п.
"""
import pandas as pd
import gzip
import json
import zipfile

import geopandas
from shapely.geometry import Point, Polygon
from planner.utils.files import resources


def load_schools(filename=resources / 'schools.zip'):
    """
    Загрузить данные о школах. Этот файл получен с data.mos.ru
    :param filename: путь до файла с информацией о школах
    :return: Geo-датафрейм с информацией о школах
    """
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
                row_data['data_type'] = 'school'

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
    """
    :param coords: Метод для получения геометрии по данным загрузочных эко-файлов
    :return: Полигон или Точка
    """
    if isinstance(coords[0], list):
        coords_data = list()
        for lat_long_row in coords:
            if len(lat_long_row) > 1:
                lat, long = lat_long_row
                coords_data.append([float(long), float(lat)])
        return Polygon(coords_data)
    else:
        lat_long_row = coords
        if len(lat_long_row) > 1:
            lat, long = lat_long_row
            long, lat = float(long), float(lat)

            return Point(long, lat)


ECO_NAMES = {
    'beach': 'beaches.json.gz',
    'gas-station': 'gas-stations.json.gz',
    'industrial-area': 'industrial-areas.json.gz',
    'nuklear-zone': 'nuklear-zones.json.gz',
    'protected-zone': 'protected-zones.json.gz',  # особо охраняемые зоны
    'snow-melting-station': 'snow-melting-stations.json.gz',
    'transport-nodes': 'transport-nodes.json.gz',
}


def load_eco_data(filename=resources / 'moskow-eco.json.gz'):
    """
    Подгрузить экоданным Москвы. Они берутся с сайта https://www.kvartiravmoskve.ru/eko/
    """
    with gzip.open(filename) as sch_stream:
        ecodata = json.load(sch_stream)

    res = pd.DataFrame(ecodata)

    del res['Color']

    del res['Image']

    res.rename(columns={'Type': 'data_type'}, inplace=True)

    geometry = res.Coords.apply(load_point_geometry)
    res = geopandas.GeoDataFrame(res, geometry=geometry)
    return res


def load_all_eco_data(filename_entries=None):
    """
    Загрузить все доступные экоданные
    """
    filename_entries = filename_entries or ECO_NAMES

    dfs = list()

    for k, v in filename_entries.items():
        df = load_eco_data(resources / v)
        df['data_type'] = k
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_buildings_data(filename=resources / 'moskow-buildings.json.gz'):
    """
    Загрузить данные строений. Данные получены с https://www.kvartiravmoskve.ru/eko/

    """
    with gzip.open(filename) as sch_stream:
        ecodata = json.load(sch_stream)

    res = pd.DataFrame(ecodata['Objects'])

    res.rename(columns={'Type': 'data_type'}, inplace=True)

    geometry = geopandas.points_from_xy(res.Lng, res.Lat)
    res['data_type'] = 'Building'
    res = geopandas.GeoDataFrame(res, geometry=geometry)

    return res


def load_highways(filename=resources / 'highways.gz.pq'):
    """
    Загрузить сохраненные данные о дорогах
    """

    return geopandas.read_parquet(filename)


def load_water(filename=resources / 'waters.gz.pq'):
    """
    Загрузить сохраненные данные о водоемах
    """
    return geopandas.read_parquet(filename)


def load_railroads(filename=resources / 'railroad.gz.pq'):
    """
    Загрузить сохраненные данные о железнодорожных путях
    """
    return geopandas.read_parquet(filename)


def load_parks(filename=resources / 'parks.gz.pq'):
    """
    Загрузить сохраненные данные о парках
    """
    return geopandas.read_parquet(filename)


def load_shapes(adm_id: int = None, filename=resources / 'all_shapes.gz.pq'):
    """
    Загрузить данные секторов по административному району
    """
    shapes_df = geopandas.read_parquet(filename)
    if adm_id is not None:
        shapes_df = shapes_df[shapes_df.adm_zid == adm_id].copy()
    return shapes_df


def load_adm_zones(adm_id: int = None, filename=resources / 'adm_zones.gz.pq'):
    """
    Загрузить данные административных районов
    """
    adm_zones_df = geopandas.read_parquet(filename)
    if adm_id is not None:
        adm_zones_df = adm_zones_df[adm_zones_df.adm_zid == adm_id].copy()
    return adm_zones_df
