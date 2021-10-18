import typing
import logging
from planner.optimization.loaders import load_schools

import pandas as pd
import geopandas
import numpy as np

import planner.optimization.model as M
from planner.utils import files
from planner.optimization.mocma import Optimizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


class ObjectFactory:
    def __init__(self, max_num_objects, proj_types, squares: typing.List[M.Square], stop_objects: M.StopObjects):
        self.max_num_objects = max_num_objects
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
        self.stop_objects: M.StopObjects = stop_objects

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

                if not is_stopped:
                    proj = self.proj_types[obj_type]
                    obj = M.TargetObject(obj_lng, obj_lat, proj['num_peoples'], proj)

                    res_objects.append(obj)
            else:
                pass
                # logging.info(f'fake object')

        return res_objects

    def dimension(self):
        return self.min_normalization.shape[0]


if __name__ == '__main__':
    num_objects = 10

    stop_distances = {
        'beach': 0.1,  # не менее 300 метров до пляжа
        'gas-station': 0.05,
        'industrial-area': 0.05,
        'nuklear-zone': 0.2,
        # 'protected-zone': 0.1,
        'snow-melting-station': 0.1,
        'transport-nodes': 0.1,
        'water': 0.1,
        'highway': 0.05,
        'Building': 0.05,
    }

    school_projects = [
        {
            'num_peoples': 600,
            'cost': 600 * 1e6,
        },
        {
            'num_peoples': 800,
            'cost': 800 * 1.2e6,
        },
        {
            'num_peoples': 1000,
            'cost': 1000 * 1.4e6,
        },
        {
            'num_peoples': 1200,
            'cost': 1200 * 1.4e6,
        },
    ]

    squares_df = geopandas.read_parquet(files.resources / 'short_shapes_geo.gz.pq')

    squares = []
    child_part = 0.2
    total_number_of_children = 0
    for num_peoples, geometry in squares_df[['customers_cnt_home', 'geometry']].values:
        centroid = geometry.centroid
        number_of_children = int(num_peoples * child_part)
        squares.append(M.Square(centroid.x, centroid.y, number_of_children))

        total_number_of_children += number_of_children

    print(f'pending place {total_number_of_children} children to schools')
    num_schools = int(np.ceil(total_number_of_children / 1000)) * 3
    stop_objects = M.StopObjects(stop_distances)

    factory = ObjectFactory(num_schools, proj_types=school_projects, squares=squares, stop_objects=stop_objects)
    schools = load_schools()

    squares_polygon = squares_df.unary_union

    required_schools = schools[schools.geometry.apply(lambda x: x.intersects(squares_polygon))]

    existed_school_objects = list()
    for school_geom, number_of_pupils in required_schools[['geometry', 'PupilsQuantity']].values:
        school_geom_centroid = school_geom.centroid
        existed_school_objects.append(M.TargetObject(school_geom_centroid.x, school_geom_centroid.y, number_of_pupils))

    existed_evaluation = M.Evaluation(squares, existed_school_objects)

    existed_evaluation_results = existed_evaluation.evaluate()

    existed_evaluation.move_data_to_squares()

    print(existed_evaluation_results)

    def evaluator(point):
        target_objects = factory.make_objects_from_point(point)
        result = M.Evaluation(squares, target_objects).evaluate()

        logging.info(f'{result}')

        return result


    optimizer = Optimizer(factory.dimension(), 2, evaluator, population_size=200, n_jobs=5)

    optimizer.run(200)

    print(len(squares))
