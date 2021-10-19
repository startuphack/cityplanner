import gzip
import io
import pathlib
import pickle

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
