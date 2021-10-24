"""
Класс для консольного запуска оптимизации
"""
import argparse
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

import planner.optimization.optimizers as O
import planner.optimization.loaders as L
import planner.utils.files as F

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-file",
        default=str(F.resources / 'excel_config.xlsx'),
        type=str,
        help="path to config file",
    )
    parser.add_argument("--adm-id", default=114, type=int, help="adm id")
    parser.add_argument("--results-path", default='results', type=str, help="path to dump data")
    parser.add_argument("--num-steps", default=300, type=int, help="number of optimization steps")
    parser.add_argument("--step-batch", default=30, type=int, help="number of steps between dump data")
    parser.add_argument("--population-size", default=200, type=int, help="mocma population size")

    args = parser.parse_args()
    logging.info(f'args = {args}')

    squares_df = L.load_shapes(adm_id=args.adm_id)

    configs = F.parse_excel_config(args.config_file)

    parameters = configs.get('configs', {})
    projects = configs['projects']
    limits = configs.get('limits')
    params = configs.get('configs', {})

    logging.info(f'projects: {projects}')
    logging.info(f'limits: {limits}')
    logging.info(f'params: {params}')

    optimizer = O.SchoolOptimizer(
        squares_df,
        configs['projects'],
        stop_distances=configs.get('limits'),
        population_size=args.population_size,
        step_batch=args.step_batch,
        params=params
    )


    optimizer.add_callbacks(O.DrawFrontCallback(args.results_path))

    try:
        optimizer.run_optimization(args.num_steps)
    except O.OptimizationStop as e:
        exit(e.error_code)
