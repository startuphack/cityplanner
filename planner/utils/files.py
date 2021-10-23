import gzip
import io
import pathlib
import pickle
import pandas as pd

import cloudpickle

resources = pathlib.Path(__file__).parent / '../../resources'

if __name__ == '__main__':
    print(list(resources.glob('*')))


def pickle_dump(object, filename, gzip_file=True):
    """
    Сохранить объект в файл
    :param object: сохраняемый объект
    :param filename: имя файла
    :param gzip_file: сжимать ли сериализацию объекта с gzip
    """
    o_method = gzip.open if gzip_file else open

    with io.BufferedWriter(o_method(filename, 'w')) as output:
        cloudpickle.dump(object, output, protocol=0)


def pickle_load(filename, gzip_file=True):
    """
    Загрузить объект из файла filename
    :param filename: имя файла
    :param gzip_file: является ли файл сжатым с gzip
    :return: объект, загруженный из файла
    """
    i_method = gzip.open if gzip_file else open
    with i_method(filename) as input:
        return pickle.load(input)


def parse_excel_config(config_file):
    projects_data = pd.read_excel(config_file, sheet_name='projects')
    projects_data.columns = map(lambda x: x.strip().lower(), projects_data.columns)
    projects_data.rename(columns={
        'проект': 'name',
        'стоимость постройки': 'cost',
        'число мест': 'num_peoples',
        'включено': 'enabled',
    }, inplace=True)

    projects_data = projects_data[projects_data.enabled.str.lower() == 'да']

    projects_dict = projects_data.to_dict(orient='records')
    try:
        limits_data = pd.read_excel(config_file, sheet_name='limits')
        limits_data.columns = map(lambda x: x.strip().lower(), limits_data.columns)
        limits_data.rename(columns={
            'id ограничения': 'lid',
            'значение ограничения(км)': 'limit',
            'включено': 'enabled',
        }, inplace=True)

        limits_data = limits_data[limits_data.enabled.str.lower() == 'да']

        limits_dict = limits_data.set_index('lid').to_dict()['limit']
    except ValueError:
        limits_dict = None

    try:
        configs_data = pd.read_excel(config_file, sheet_name='configs')
        configs_data.columns = map(lambda x: x.strip().lower(), configs_data.columns)
        configs_data.rename(columns={
            'параметр': 'key',
            'значение': 'value'
        }, inplace=True)

        configs_data = configs_data.set_index('key').to_dict()['value']

    except ValueError:
        configs_data = None

    result = {
        'projects': projects_dict,
    }

    if limits_dict:
        result['limits'] = limits_dict

    if configs_data:
        result['configs'] = configs_data

    return result
