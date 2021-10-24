"""
Это тестовый файл для отладки оптимизации. Здесь представлены все основные модули и параметры.
"""
import logging

import geopandas


import planner.optimization.optimizers as O
from planner.utils import files

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


def fn():
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

    optimizer = O.SchoolOptimizer(squares_df, school_projects, population_size=200, step_batch = 10)
    optimizer.add_callbacks(O.DrawFrontCallback(files.resources/'opt'))

    optimizer.run_optimization(200)


if __name__ == '__main__':
    fn()
