from shapely.geometry import Point
import pandas as pd
import typing
from math import radians
import logging

import numpy as np
import pandas as pd

from sklearn.neighbors import BallTree
from shapely.strtree import STRtree

from shapely.geometry import Point
import planner.optimization.loaders as L
from planner.utils.geo_utils import EARTH_SIZE, find_buffer_width
from planner.utils.math import weighted_percentile


class Square:
    """
    Данные сектора
    """
    def __init__(self, lon, lat, num_peoples, **attrs):
        self.lon = lon
        self.lat = lat
        self.num_peoples = num_peoples
        self.attrs = dict(attrs)
        self.attrs['full_num_peoples'] = num_peoples

    def coords(self):
        return [self.lat, self.lon]

    def radian_coords(self):
        return [radians(self.lat), radians(self.lon)]


class TargetObject(Square):
    """
    Данные целевого объекта.
    Содержат кроме всего прочего информацию о проекте застройки
    """
    def __init__(self, lon, lat, num_peoples, project=None, **attrs):
        super().__init__(lon, lat, num_peoples, **attrs)
        self.project = project or {'num_peoples': num_peoples}


class Evaluation:
    """
    Состояние оценки метрик.
    Во время оценки производится попытка разместить всех потенциальных учеников в ближайшие доступные школы
    После этого считаются метрики по стоимости/удобству
    """

    def __init__(self,
                 squares: typing.List[Square],
                 objects: typing.List[TargetObject],
                 query_objects=5,
                 percentiles_for_evaluation=(50, 70, 90, 95),
                 lack_penalty=10,
                 ):
        self.squares = squares
        self.objects = objects
        self.query_objects = query_objects
        self.percentiles_for_evaluation = percentiles_for_evaluation
        self.cost_normalization = 1e10  # делим на 10 млрд
        self.lack_penalty = lack_penalty

        self.localization = {
            'total-cost': {'title': 'стоимость, млрд', 'multiplier': -self.cost_normalization / 1e9},
            'convenience': {'title': 'неудобство', 'multiplier': 1},
            'avg-distance': {'title': 'среднее расстояние, км', 'multiplier': -1},
            'overplanned': 'излишек мест',
            'lack': 'нехватка мест',
            'lack-percent': 'нехватка, %',
            'required': 'необходимо метст',
            'planned': 'запланировано мест',
            'number-of-objects': 'число объектов',
            'usability':'удобство, %'
        }

        for p in percentiles_for_evaluation:
            self.localization[f'distance.p={p}'] = F'перцентиль({p}) расстояния до объекта'

        self.obj_num_index = {
            id: {
                'obj': obj,
                'available': obj.num_peoples,
                'square-links': list(),
            }
            for id, obj in enumerate(self.objects)
        }

        self.squares_data = [
            {

                'obj': square,
                'num_peoples': square.num_peoples,
            }
            for square in self.squares
        ]

        self.reindex()

    def move_data_to_squares(self):
        for square, square_data in zip(self.squares, self.squares_data):
            square.num_peoples = square_data['num_peoples']

    def reindex(self):
        # Для ускорения оценки используем BallTree индекс. Он умеет индексировать объекты по haversine метрике.
        # Индексируем только то, что нам нужно
        target_pairs = [
            (obj_num, o.radian_coords()) for obj_num, o in enumerate(self.objects)
            if self.obj_num_index[obj_num]['available'] > 0
        ]

        if target_pairs:
            numbers, target_objects = zip(*target_pairs)
            target_objects = np.asarray(target_objects)
            self.obj_geo_index = BallTree(target_objects, metric='haversine')
            self.idx_to_num = dict(enumerate(numbers))
        else:
            self.obj_geo_index = None
            self.idx_to_num = dict()

    def get_metrics(self):
        """
        Здесь производится подсчет результата размещения учеников по школам.
        По итогам возвращаются полученные метрики
        """
        overplanned = 0
        all_planned = 0
        distances, weights = list(), list()
        total_cost = 0

        all_required = sum(place.num_peoples for place in self.squares)

        for target_object_data in self.obj_num_index.values():

            target_object = target_object_data['obj']
            all_planned += target_object.num_peoples

            cost = target_object.project.get('cost', 0)
            total_cost += cost
            overplanned += target_object_data['available']
            for square_link in target_object_data['square-links']:
                # square = square_link['square']
                placed = square_link['placed']
                idx_distance = square_link['distance']
                idx_distance_km = idx_distance * EARTH_SIZE

                distances.append(idx_distance_km)
                weights.append(placed)

        distances = np.asarray(distances)
        weights = np.asarray(weights)

        if weights.sum() > 0:
            avg_distance = (distances * weights).sum() / weights.sum()
            percentiles = weighted_percentile(distances, self.percentiles_for_evaluation, w=weights)
        else:
            avg_distance = 1e9
            percentiles = [np.inf] * len(self.percentiles_for_evaluation)

        lack = max(all_required - all_planned, 0)
        lack_percent = lack / all_required
        lack_penalty = self.lack_penalty * lack_percent
        lack_penalty *= lack_penalty

        result = {
            'total-cost': -total_cost / self.cost_normalization,
            'convenience': -(avg_distance + lack_penalty),
            'avg-distance': -avg_distance,
            'overplanned': overplanned,
            'lack': lack,
            'lack-percent': lack_percent,
            'required': all_required,
            'planned': all_planned,
            'number-of-objects': len(self.objects)
        }

        for p_value, p in zip(percentiles, self.percentiles_for_evaluation):
            result[f'distance.p={p}'] = p_value

        return result

    def evaluate(self):
        """
        Разместить учеников по школам.
        В каждый момент выбираются ближайшие по haversine метрике школы.
        В них размещается столько учеников, сколько есть в секторе, либо есть свободных мест в школах.
        Когда места в школах заканчиваются, производится переиндексация доступных школ.
        """
        reindex_required = False
        no_objects = False

        if self.objects:
            for square, square_geo in zip(self.squares_data, self.squares):

                if no_objects:
                    break

                if square['num_peoples'] > 0:
                    required_to_place = square['num_peoples']
                    first_pass = True
                    while True:
                        if (required_to_place == 0) or ((not first_pass) and (not reindex_required)):
                            # Мы останавливаемся, если
                            #   полностью исчерпали текущий квадрат
                            #   или у нас не исчерпались новые целевые объекты (это значит, что у нас уже все исчерпано)
                            break

                        first_pass = False
                        if reindex_required:
                            self.reindex()
                            reindex_required = False
                            if not self.idx_to_num:
                                no_objects = True
                                break

                        objects_to_query = min(len(self.idx_to_num), self.query_objects)

                        distances, obj_nums = self.obj_geo_index.query([square_geo.radian_coords()], objects_to_query)

                        if len(obj_nums) == 0:
                            no_objects = True
                            break

                        for distance, idx_obj_num in zip(distances.flatten(), obj_nums.flatten()):
                            obj_num = self.idx_to_num[idx_obj_num]
                            obj_info = self.obj_num_index[obj_num]
                            available = obj_info['available']
                            if available > 0:
                                can_place = min(available, required_to_place)
                                if can_place == available:
                                    reindex_required = True

                                obj_info['square-links'].append(
                                    {
                                        'square': square,
                                        'placed': can_place,
                                        'distance': distance,
                                    }
                                )

                                obj_info['available'] -= can_place
                                required_to_place -= can_place
                                square['num_peoples'] -= can_place

                                if required_to_place == 0:
                                    break

        return self.get_metrics()


class StopObjects:
    '''
    Класс для учета недоступных для размещения территорий.
    Каждая недоступная для размещения территория окружается окрестностью (ширина окрестности указывается в параметрах)
    После чего полигон вместе с окрестностью считается невозможным для строительства.
    Попавшие в полигон объекты считаются непостроенными.
    Для ускорения расчета используется R-tree дерево.
    Окрестность задается в километрах, после чего пересчитывается в дельта по широте, дельта по долготе.


    ВНИМАНИЕ!!!
    во всех геоданных сначала идет долгота, а потом широта. для haversine_distance нужно наоборот!!!
    для проверки нужно менять порядок широты с долготой
    '''
    stop_loaders = [
        L.load_all_eco_data,
        L.load_water,
        L.load_highways,
        L.load_buildings_data,
        L.load_parks,
        L.load_railroads,
    ]

    def __init__(self, stop_distance_for_objects):
        self.stop_distance_for_objects = stop_distance_for_objects
        self.geo_tree = None
        self.geo_data = None
        self.build_stop_index()

    def build_stop_index(self):
        geometry_for_index = list()
        geo_points_data = list()
        for loader in self.stop_loaders:
            df = loader()
            for data_type in df.data_type.unique():
                stop_distance = self.stop_distance_for_objects.get(data_type.lower())

                if stop_distance:
                    data_geometry = df[df.data_type == data_type].geometry
                    centroid = data_geometry.centroid
                    stop_distance_in_angles = find_buffer_width(
                        [np.median(centroid.y), np.median(centroid.x)],
                        stop_distance)

                    logging.info(f'stop distance = {stop_distance} km; angles = {stop_distance_in_angles} for {data_type}')
                    buffered_data_geometry = data_geometry.buffer(stop_distance_in_angles)
                    geometry_for_index.append(buffered_data_geometry)
                    geo_points_data.append({
                        'data_type': data_type,
                        'distance': stop_distance,
                        'distance_in_angles': stop_distance_in_angles,
                    })
                else:
                    logging.info(f'no stop distance for {data_type}')

        geo_points = []
        for points_data, geometry in zip(geo_points_data, geometry_for_index):
            geo_points += [points_data] * len(geometry)

        self.merged_geometry = pd.concat(geometry_for_index, ignore_index=True)
        # self.geo_tree = self.merged_geometry.sindex
        self.geo_tree = STRtree(self.merged_geometry)
        self.geo_data = np.asarray(geo_points)

    def is_stopped(self, lat, lon):
        intersection_result = self.geo_tree.query(Point(lon, lat))

        return intersection_result


class ObjectFactory:
    """
    Класс, который отвечает за генерацию набора целевых объектов по данным вектора.
    """
    def __init__(self, max_num_objects, proj_types, squares: typing.List[Square], stop_objects: StopObjects, squares_polygon = None):
        self.max_num_objects = max_num_objects
        self.squares_polygon = squares_polygon
        self.proj_types = proj_types
        self.squares = squares

        lats, longs = zip(*(s.coords() for s in squares))
        self.min_lat, self.max_lat = min(lats), max(lats)
        self.min_lng, self.max_lng = min(longs), max(longs)

        min_normalization, max_normalization = list(), list()

        for _ in range(self.max_num_objects):
            min_normalization.extend([0, self.min_lat, self.min_lng])
            max_normalization.extend([len(proj_types) + 4 - 1e-6, self.max_lat, self.max_lng])

        self.min_normalization = np.asarray(min_normalization)
        self.max_normalization = np.asarray(max_normalization)
        self.stop_objects: StopObjects = stop_objects

    def make_objects_from_point(self, point: np.ndarray):

        normalized_array = point * (self.max_normalization - self.min_normalization) + self.min_normalization
        res_objects = list()

        idx = 0
        for _ in range(self.max_num_objects):
            obj_type = int(normalized_array[idx])
            idx += 1
            obj_lat = normalized_array[idx]
            idx += 1
            obj_lng = normalized_array[idx]
            idx += 1

            if obj_type < len(self.proj_types):
                is_stopped = False
                if self.stop_objects:
                    is_stopped = bool(self.stop_objects.is_stopped(obj_lat, obj_lng))

                intersects = (self.squares_polygon is None) or (self.squares_polygon.intersects(Point(obj_lng, obj_lat)))

                if not is_stopped and intersects:
                    proj = self.proj_types[obj_type]
                    obj = TargetObject(obj_lng, obj_lat, proj['num_peoples'], proj)

                    res_objects.append(obj)
            else:
                pass
                # logging.info(f'fake object')

        return res_objects

    def dimension(self):
        return self.min_normalization.shape[0]
