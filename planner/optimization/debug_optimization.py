import typing
import logging

import pandas as pd
import numpy as np

import planner.optimization.model as M
from planner.utils import files
from planner.optimization.mocma import Optimizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


class ObjectFactory:
    def __init__(self, max_num_objects, proj_types, squares: typing.List[M.Square]):
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

    squares_df = pd.read_parquet(files.resources / 'short_shapes.gz.pq')

    squares = []
    child_part = 0.2
    total_number_of_children = 0
    for lat, lon, num_peoples in squares_df[['lat', 'lon', 'customers_cnt_home']].values:
        number_of_children = int(num_peoples * child_part)
        squares.append(M.Square(lon, lat, number_of_children))

        total_number_of_children += number_of_children

    print(f'pending place {total_number_of_children} children to schools')
    num_schools = int(np.ceil(total_number_of_children / 1000)) * 2

    factory = ObjectFactory(num_schools, proj_types=school_projects, squares=squares)


    def evaluator(point):
        target_objects = factory.make_objects_from_point(point)
        result = M.Evaluation(squares, target_objects).evaluate()

        logging.info(f'{result}')

        return result


    optimizer = Optimizer(factory.dimension(), 2, evaluator, population_size=200, n_jobs = 5)

    optimizer.run(200)

    print(len(squares))
