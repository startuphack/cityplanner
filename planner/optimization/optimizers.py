import logging
import math
import os
import typing

import numpy as np
import pandas as pd
import plotly.express as px
from shapely.geometry import Point
import planner.optimization.model as M
from planner.optimization.loaders import load_schools
from planner.optimization.mocma import Optimizer
from planner.utils.files import pickle_dump
from planner.utils.math import is_pareto_efficient
from sklearn.preprocessing import QuantileTransformer

STOP_DISTANCES = {
    'beach': 0.1,  # не менее 100 метров до пляжа
    'gas-station': 0.05,
    'industrial-area': 0.05,
    'nuklear-zone': 0.2,
    # 'protected-zone': 0.01,
    'snow-melting-station': 0.1,
    'transport-nodes': 0.1,
    'water': 0.1,
    'highway': 0.05,
    'park': 0.001,
    'building': 0.05,
}


class OptimizationStop(Exception):
    error_code: int = NotImplemented


class NoSchoolsRequired(OptimizationStop):
    error_code = 101


class OptimizationCallback:
    def on_step(self, algorithm, optimizer):
        """

        :param algorithm: MOCMA-ES
        :param optimizer: SchoolOptimizer
        :return: True, если продолжаем оптимизацию
        """
        return True


class SchoolOptimizer:
    """
    Оптимизатор расположения школ. Собирает все параметры вместе, запускает оптимизацию, сохраняет результаты
    """

    def __init__(self,
                 squares_df,
                 school_projects,
                 stop_distances=None,
                 step_batch=30,
                 population_size=200,
                 params=None
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
        self.params = params or {}

        self.population_size = population_size
        self.school_child_percent = self.params.get('pupilsPerCitizen', 0.11)
        self.lack_penalty_coefficient = self.params.get('lackPenaltyCoefficient', 4.9)

    def add_callbacks(self, *callbacks):
        self.callbacks.extend(callbacks)

    def __getstate__(self):
        return {
            'factory': self.factory,
            'squares': self.squares,
        }  # only this is needed

    def _init_optimization(self):
        squares = []

        # Считаем общее кол-во потенциальных учеников
        total_number_of_children = 0
        for num_peoples, geometry in self.squares_df[['customers_cnt_home', 'geometry']].values:
            centroid = geometry.centroid
            number_of_children = int(num_peoples * self.school_child_percent)
            squares.append(M.Square(centroid.x, centroid.y, number_of_children))

            total_number_of_children += number_of_children

        print(f'pending place {total_number_of_children} children to schools')

        # В среднем школа на 1000 учеников. Делаем 3x запас для различных вариантов размещения
        num_schools = int(np.ceil(total_number_of_children / 1000)) * 3
        stop_objects = M.StopObjects(self.stop_distances)

        squares_polygon = self.squares_df.unary_union

        factory = M.ObjectFactory(
            num_schools,
            proj_types=self.school_projects,
            squares=squares,
            stop_objects=stop_objects,
            squares_polygon = squares_polygon,
        )
        schools = load_schools()

        required_schools = schools[schools.geometry.apply(lambda x: x.intersects(squares_polygon))]
        # Учитываем текущие построенные школы и данные об их учениках.
        # Исключаем текущих размещенных в школы учеников из необходимых для размещения
        existed_school_objects = list()
        for school_geom, number_of_pupils in required_schools[['geometry', 'PupilsQuantity']].values:
            school_geom_centroid = school_geom.centroid
            existed_school_objects.append(
                M.TargetObject(school_geom_centroid.x, school_geom_centroid.y, number_of_pupils))

        existed_evaluation = M.Evaluation(squares, existed_school_objects)

        existed_evaluation_results = existed_evaluation.evaluate()
        # Обновляем данные секторов с учетом текущих построенных школ (Data.mos.ru)
        existed_evaluation.move_data_to_squares()

        remained_number_of_pupils = sum(square.num_peoples for square in squares)

        if remained_number_of_pupils == 0:
            self.no_schools_required = True
            logging.info(f'no required new schools')
        else:
            self.no_schools_required = False

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
        """
        Шаг за шагом выполняем оптимизацию
        Через каждые self.step_batch шагов запускаем слушателей.
        """
        if self.optimizer is None:
            self._init_optimization()
            # Если школы не нужны, не оптимизируем
            if self.no_schools_required:
                logging.info(f'no school required')
                raise NoSchoolsRequired

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
    """
    Этот callback отвечает за сохранение данных, по которым восстанавливаются точки парето-фронта.
    """

    def __init__(self, target_path='.'):
        self.target_path = target_path
        os.makedirs(target_path, exist_ok=True)

    def on_step(self, algorithm, optimizer):
        history_data = algorithm.history()

        points, metrics = zip(*history_data)

        # Убираем заведомо неэффективные точки
        points_df = pd.DataFrame(map(lambda x: x.metrics, metrics))
        points_df['usability'] = 100 * QuantileTransformer().fit_transform(
            points_df['convenience'].values.reshape(-1, 1))[:, 0]
        low_convenience = np.percentile(points_df['convenience'], q=50)
        low_cost = np.percentile(points_df['total-cost'], q=50)
        predicate = (points_df.convenience > low_convenience) | (points_df['total-cost'] > low_cost)
        predicate = predicate & (points_df['convenience'] > -1e2)
        predicate = predicate & (points_df['total-cost'] < 0)
        predicate = predicate & (points_df['avg-distance'] > - 5)

        points_df = points_df[predicate].copy()
        points = np.asarray(points)[predicate]

        pareto_data = points_df[['convenience', 'total-cost']].values
        # Оставляем только эффективные по парето точки
        pareto_mask = is_pareto_efficient(pareto_data)

        points_df = points_df[pareto_mask]
        points = points[pareto_mask]

        points_df['point_id'] = points_df.index
        points_df['point'] = list(points)

        plot_df = points_df.sort_values(['convenience'], ascending=False)

        plot_df_data = plot_df.to_dict(orient='records')
        dump_data = {
            'plot_data': plot_df_data,
            'factory': optimizer.factory,
            'squares': optimizer.squares,
        }

        pickle_dump(dump_data, f'{self.target_path}/pareto_{algorithm.fitness_steps}.gz.mdl')

        point_col = plot_df['point']
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
        plot_df = plot_df.round(2)

        fig = px.scatter(plot_df, x='стоимость, млрд', y='среднее расстояние, км',
                         color='удобство, %', hover_data=plot_df.columns, size='size')
        #
        fig.write_html(f'{self.target_path}/pareto_{algorithm.fitness_steps}.html')

        plot_df['point'] = point_col
        pickle_dump(
            {'plot_df': plot_df, 'factory': optimizer.factory},
            f'{self.target_path}/step_{algorithm.fitness_steps}.gz.mdl',
        )
