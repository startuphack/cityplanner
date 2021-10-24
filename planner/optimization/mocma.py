"""
В этом файле находится модуль многокритериальной оптимизации
"""
import csv
import logging
import typing

import itertools as it
import math
import more_itertools as mit
import numpy as np
from deap import base
from deap import cma
from deap import creator
from deap import tools
from joblib.parallel import Parallel, delayed
from runstats import Statistics


def do_evaluate(evaluator, batch, batch_evaluation=False):
    """
    Метод, отвечающий за получение метрик по выбранным точкам
    :param evaluator: функция оценки
    :param batch: точка или набор точек для получения метрик
    :param batch_evaluation: флаг того, что оценка производится пакетно. это используется для оптимизации вычислений
    :return: метрики для точки/набора точек
    """
    if batch_evaluation:
        indices, values = zip(*batch)
        results = evaluator(values)
        return list(zip(indices, results))
    else:
        result = []
        for ind, v in batch:
            result.append((ind, evaluator(v)))

        return result


class OptResults:
    """
    Обертка для получения дополнительных метрик в момент оценки.
    Целевыми метриками, влияющими на оптимизацию считаются первые dim штук
    """
    def __init__(self, dim, values):
        if isinstance(values, dict):
            self.values = [v for k, v in it.islice(values.items(), dim)]
            self.metrics = values
        else:
            self.values = values[:dim]
            self.metrics = {idx: v for idx, v in enumerate(values)}

    def __repr__(self):
        return str(self.metrics)


class OptFitness(base.Fitness):
    """
    Один представитель поколения оптимизации
    """
    def __init__(self, values=()):
        super().__init__(values)
        self.results = None

    def set_results(self, results: OptResults):
        self.values = results.values
        self.results = results


DEFAULT_BACKEND = Parallel(n_jobs=4, backend='multiprocessing')


class Optimizer:
    def __init__(self,
                 problem_size,
                 num_objectives,
                 optimization_target,
                 fitness_weights=None,
                 min=0,
                 max=1,
                 population_size=100,
                 mutation_percent=1,
                 sigma=None,
                 backend=DEFAULT_BACKEND,
                 min_batch_size=1,
                 batch_evaluation=False,
                 round_digits=None
                 ):
        self.problem_size = problem_size
        self.num_objectives = num_objectives
        self.optimization_target = optimization_target
        self.fitness_weights = fitness_weights or ([1] * num_objectives)
        self.min_value = min
        self.max_value = max
        self.min = np.full(problem_size, min)
        self.max = np.full(problem_size, max)
        self.delta = self.max - self.min
        self.population_size = population_size  # mu
        self.mutation_percent = mutation_percent  # lambda
        self.sigma = sigma or (self.max - self.min) / 3
        self.fitness_history = []
        self.fitness_steps = 0
        self.evaluations = {}
        self.toolbox = None
        self.batch_evaluation = batch_evaluation
        self.backend = backend
        self.min_batch_size = min_batch_size
        self.round_digits = round_digits

    def __getstate__(self):
        return {}  # only this is needed

    def normalize(self, individual):
        """
        Нормализовать передаваемые данные, чтобы они попадали в интервал [self.min, self.max]
        :param individual: вектор одной точки
        :return: номрализованный вектор
        """
        new_val = self.min + self.delta * ((1 - np.sin(individual)) / 2)
        if self.round_digits is not None:
            new_val = np.round(new_val, self.round_digits)
        return new_val

    def v_key(self, v):
        """
        Мы кешируем точки с помощью нормализованного строкового представления
        Этот метод используется для получения ключа
        """
        return v.tostring()

    def on_step_complete(self):
        for listener in self.step_listeners:
            listener(self)

    def evaluate(self, vectors):
        """
        В этом методе происходит вычисление метрик для всего поколения точек.
        Вычисления происходят в пуле процессов и, при необходимости, кешируются
        :return: Значения метрик для поколения
        """
        result = [None] * vectors.shape[0]

        remains = len(vectors)

        for ind, v in enumerate(vectors):
            evaluation = self.evaluations.get(self.v_key(v))
            if evaluation is not None:
                result[ind] = evaluation
                remains -= 1

        n_jobs = self.backend.n_jobs
        batch_size = max(int(math.ceil(remains / n_jobs)), self.min_batch_size)
        for_evaluation = ((ind, v) for ind, v in enumerate(vectors) if result[ind] is None)

        if remains > 0:
            evaluated_values = self.backend(
                delayed(do_evaluate)(self.optimization_target, batch, self.batch_evaluation) for batch in
                mit.chunked(for_evaluation, batch_size)
            )

            for ind, val in it.chain.from_iterable(evaluated_values):
                result[ind] = OptResults(self.num_objectives, val)

            for v, val in zip(vectors, result):
                self.evaluations[self.v_key(v)] = val

        return result

    def fit_population(self, population):
        """
        Заполнить метрики для поколения
        :param population: поколения
        """

        vectors = np.asarray(list(map(self.normalize, population)))

        for ind, objectives in zip(population, self.evaluate(vectors)):
            fit = objectives
            ind.fitness.set_results(fit)
            self.fitness_history.append(fit.values)

    def step(self):
        """
        Шаг оптимизации.
        Состоит из
        1. Генерации наследников
        2. Оценка метрик
        3. Обновление состояния алгоритма
        """
        logging.info(f'optimization step = {self.fitness_steps + 1}')
        # Generate a new population
        population = self.toolbox.generate()

        self.fit_population(population)

        # Update the strategy with the evaluated individuals
        self.toolbox.update(population)

        self.fitness_steps += 1

    def do_init(self):
        """
        Инициализация состояния алгоритма.
        Инициализируем случайными точками
        """
        if self.toolbox is not None:
            return

        creator.create('FitnessMin', OptFitness, weights=self.fitness_weights)
        creator.create('Individual', list, fitness=creator.FitnessMin)
        # The MO-CMA-ES algorithm takes a full population as argument
        self.toolbox = base.Toolbox()

        population = [
            creator.Individual(x) for x in
            (np.random.uniform(self.min_value, self.max_value, (self.population_size, self.problem_size)))
        ]

        self.fit_population(population)

        strategy = cma.StrategyMultiObjective(
            population,
            sigma=self.sigma,
            mu=self.population_size,
            lambda_=int(self.population_size * self.mutation_percent)
        )

        self.toolbox.register('generate', strategy.generate, creator.Individual)
        self.toolbox.register('update', strategy.update)
        self.strategy = strategy

    def run(self, num_steps):
        self.do_init()
        for _ in range(num_steps):
            self.step()

    def pareto(self):
        """
        Получить текущий парето-фронт алгоритма
        """
        return [
            (self.normalize(ind), ind.fitness.results)
            for ind in self.strategy.parents
        ]

    def history(self) -> typing.Iterable[typing.Tuple[np.array, OptResults]]:
        """
        Получить всю историю вычислений
        """
        for point_str, results in self.evaluations.items():
            point = np.frombuffer(point_str, np.float64)
            yield point, results
