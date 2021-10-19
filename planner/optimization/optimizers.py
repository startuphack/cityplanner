import pandas as pd
import logging
import os
import math
import typing
import plotly.express as px
import numpy as np

import planner.optimization.model as M
from planner.optimization.loaders import load_schools
from planner.optimization.mocma import Optimizer
from sklearn.cluster import MiniBatchKMeans
from planner.utils.files import pickle_dump

import plotly

STOP_DISTANCES = {
    'beach': 0.1,  # не менее 300 метров до пляжа
    'gas-station': 0.05,
    'industrial-area': 0.05,
    'nuklear-zone': 0.2,
    # 'protected-zone': 0.01,
    'snow-melting-station': 0.1,
    'transport-nodes': 0.1,
    'water': 0.1,
    'highway': 0.05,
    'Building': 0.05,
}


class OptimizationCallback:
    def on_step(self, algorithm, optimizer):
        """

        :param algorithm: MOCMA-ES
        :param optimizer: SchoolOptimizer
        :return: True, если продолжаем оптимизацию
        """
        return True


class SchoolOptimizer:
    def __init__(self,
                 squares_df,
                 school_projects,
                 stop_distances=None,
                 step_batch=30,
                 population_size=200,
                 school_child_percent=0.2,
                 ):
        self.squares_df = squares_df
        self.school_projects = school_projects
        self.stop_distances = stop_distances or STOP_DISTANCES
        self.callbacks: typing.List[OptimizationCallback] = list()
        self.factory = None
        self.optimizer: Optimizer = None
        self.required_schools = None
        self.squares = None
        self.step_batch = step_batch

        self.population_size = population_size
        self.school_child_percent = school_child_percent

    def add_callbacks(self, *callbacks):
        self.callbacks.extend(callbacks)

    def __getstate__(self):
        return {
            'factory': self.factory,
            'squares': self.squares,
        }  # only this is needed

    def _init_optimization(self):
        squares = []

        total_number_of_children = 0
        for num_peoples, geometry in self.squares_df[['customers_cnt_home', 'geometry']].values:
            centroid = geometry.centroid
            number_of_children = int(num_peoples * self.school_child_percent)
            squares.append(M.Square(centroid.x, centroid.y, number_of_children))

            total_number_of_children += number_of_children

        print(f'pending place {total_number_of_children} children to schools')
        num_schools = int(np.ceil(total_number_of_children / 1000)) * 3
        stop_objects = M.StopObjects(STOP_DISTANCES)

        factory = M.ObjectFactory(num_schools, proj_types=self.school_projects, squares=squares,
                                  stop_objects=stop_objects)
        schools = load_schools()

        squares_polygon = self.squares_df.unary_union

        required_schools = schools[schools.geometry.apply(lambda x: x.intersects(squares_polygon))]

        existed_school_objects = list()
        for school_geom, number_of_pupils in required_schools[['geometry', 'PupilsQuantity']].values:
            school_geom_centroid = school_geom.centroid
            existed_school_objects.append(
                M.TargetObject(school_geom_centroid.x, school_geom_centroid.y, number_of_pupils))

        existed_evaluation = M.Evaluation(squares, existed_school_objects)

        existed_evaluation_results = existed_evaluation.evaluate()

        existed_evaluation.move_data_to_squares()

        logging.info(f'current schools data: {existed_evaluation_results}')

        optimizer = Optimizer(factory.dimension(), 2, self.evaluator, population_size=self.population_size)

        self.factory = factory
        self.optimizer = optimizer
        self.required_schools = required_schools
        self.squares = squares

    def evaluator(self, point):
        target_objects = self.factory.make_objects_from_point(point)
        result = M.Evaluation(self.squares, target_objects).evaluate()

        logging.info(f'{result}')

        return result

    def evaluation(self):
        return M.Evaluation(self.squares, [])

    def run_optimization(self, num_steps=1000):
        if self.optimizer is None:
            self._init_optimization()

        required_steps = int(math.ceil(num_steps / self.step_batch))

        for step in range(required_steps):
            if self.optimizer.fitness_steps:
                for _ in range(self.step_batch):
                    self.optimizer.step()
            else:
                self.optimizer.run(self.step_batch)

            do_continue_optimization = True
            for callback in self.callbacks:
                do_continue = callback.on_step(self.optimizer, self)

                do_continue_optimization = do_continue_optimization and (not (do_continue is False))

            if not do_continue_optimization:
                break


class DrawFrontCallback(OptimizationCallback):
    def __init__(self, target_path='.'):
        self.target_path = target_path
        os.makedirs(target_path, exist_ok=True)

    def on_step(self, algorithm, optimizer):
        history_data = algorithm.history()

        points, metrics = zip(*history_data)

        points_df = pd.DataFrame(map(lambda x: x.metrics, metrics))
        low_convenience = np.percentile(points_df['convenience'], q=90)
        low_cost = np.percentile(points_df['total-cost'], q=90)
        predicate = (points_df.convenience > low_convenience) | (points_df['total-cost'] > low_cost)
        points_df = points_df[predicate].copy()
        points = np.asarray(points)[predicate]

        pareto_data = points_df[['convenience', 'total-cost']].values

        point_labels = MiniBatchKMeans(n_clusters=min(len(points_df), 10)).fit_predict(pareto_data)

        points_df['cluster'] = point_labels
        points_df['point_id'] = points_df.index
        points_df['point'] = list(points)

        plot_df = points_df.sort_values(['cluster', 'convenience'], ascending=False)

        plot_df_data = plot_df.to_dict(orient='records')
        dump_data = {
            'plot_data': plot_df_data,
            'factory': optimizer.factory,
            'squares': optimizer.squares,
        }

        pickle_dump(dump_data, f'{self.target_path}/pareto_{algorithm.fitness_steps}.gz.mdl')

        plot_df.drop_duplicates(['cluster'], inplace=True)

        del plot_df['point']

        target_columns = list()
        for k, v in optimizer.evaluation().localization.items():
            if isinstance(v, dict):
                title = v['title']
                multiplier = v['multiplier']
            else:
                title = v
                multiplier = 1

            target_columns.append(title)
            plot_df[title] = plot_df[k] * multiplier

        plot_df = plot_df[target_columns]
        plot_df['size'] = 2

        fig = px.scatter(plot_df, x='стоимость, млрд', y='среднее расстояние',
                         color='неудобство', hover_data=plot_df.columns, size='size')
        #
        fig.write_html(f'{self.target_path}/pareto_{algorithm.fitness_steps}.html')
