import typing
from math import radians
import logging

import numpy as np
import pandas as pd
import pygeos
from sklearn.metrics.pairwise import haversine_distances
from sklearn.neighbors import BallTree

import planner.optimization.loaders as L
from planner.utils.geo_utils import EARTH_SIZE, find_buffer_width
from planner.utils.math import weighted_percentile


class Square:
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
    def __init__(self, lon, lat, num_peoples, project=None, **attrs):
        super().__init__(lon, lat, num_peoples, **attrs)
        self.project = project or {'num_peoples': num_peoples}


class Evaluation:
    def __init__(self,
                 squares: typing.List[Square],
                 objects: typing.List[TargetObject],
                 query_objects=5,
                 percentiles_for_evaluation=(50, 70, 90, 95)
                 ):
        self.squares = squares
        self.objects = objects
        self.query_objects = query_objects
        self.percentiles_for_evaluation = percentiles_for_evaluation

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
                square = square_link['square']
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
        lack_penalty = 10 * lack_percent
        lack_penalty *= lack_penalty

        result = {
            'total-cost': -total_cost / 10e9,
            'convenience': -(avg_distance + lack_penalty),
            'avg-distance': -avg_distance,
            'overplanned': overplanned,
            'lack': lack,
            'lack-percent': lack_percent,
            'required': all_required,
            'planned': all_planned,
            'number-of-schools': len(self.objects)
        }

        for p_value, p in zip(percentiles, self.percentiles_for_evaluation):
            result[f'distance.p={p}'] = p_value

        return result

    def evaluate(self):
        reindex_required = False
        no_objects = False

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
    ВНИМАНИЕ!!!
    во всех геоданных сначала идет долгота, а потом широта. для haversine_distance нужно наоборот!!!
    для проверки нужно менять порядок широты с долготой
    '''
    stop_loaders = [
        L.load_all_eco_data,
        L.load_water,
        L.load_highways,
        L.load_buildings_data,
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
                stop_distance = self.stop_distance_for_objects.get(data_type)

                if stop_distance:
                    logging.info(f'stop distance = {stop_distance} km for {data_type}')
                else:
                    logging.info(f'no stop distance for {data_type}')
                if stop_distance:
                    data_geometry = df[df.data_type == data_type].geometry
                    centroid = data_geometry.centroid
                    stop_distance_in_angles = find_buffer_width(
                        [np.median(centroid.y), np.median(centroid.x)],
                        stop_distance)
                    buffered_data_geometry = data_geometry.buffer(stop_distance_in_angles)
                    geometry_for_index.append(buffered_data_geometry)
                    geo_points_data.append({
                        'data_type': data_type,
                        'distance': stop_distance,
                        'distance_in_angles': stop_distance_in_angles,
                    })

        geo_points = []
        for points_data, geometry in zip(geo_points_data, geometry_for_index):
            geo_points += [points_data] * len(geometry)

        self.merged_geometry = pd.concat(geometry_for_index, ignore_index=True)
        self.geo_tree = self.merged_geometry.sindex
        self.geo_data = np.asarray(geo_points)

    def is_stopped(self, lat, lon):
        intersection_result = self.geo_tree.intersection([lon, lat])

        try:
            all_result = [
                {
                    'merged_data': m_data,
                    'geom': geom,
                }
                for m_data, geom in zip(self.geo_data[intersection_result], self.merged_geometry[intersection_result])
            ]
        except:

            raise

        return all_result


if __name__ == '__main__':
    bsas = [-34.83333, -58.5166646]
    paris = [49.0083899664, 2.53844117956]

    bsas_in_radians = [radians(_) for _ in bsas]
    paris_in_radians = [radians(_) for _ in paris]
    result = haversine_distances([bsas_in_radians, paris_in_radians])

    # from sklearn.neighbors import BallTree
    # # print (BallTree.valid_metrics)
    # import numpy as np
    #
    # X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    #
    # # import timeit
    # # timeit.repeat(lambda x: BallTree(X, leaf_size=30, metric='haversine'))
    # kdt = BallTree(X, leaf_size=30, metric='haversine')
    # print(kdt.query(X[:1], k=5, return_distance=True, sort_results=True))
    # # array([[0, 1],
    # #        [1, 0],
    # #        [2, 1],
    # #        [3, 4],
    # #        [4, 3],
    # #        [5, 4]]...)
