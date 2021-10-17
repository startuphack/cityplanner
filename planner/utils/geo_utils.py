import numpy as np
from scipy.optimize import newton_krylov
from sklearn.metrics.pairwise import haversine_distances

EARTH_SIZE = 6371  # km


def haver_distance(radian_coords1, radian_coords2):
    '''
    Координаты передаются в виде широта, долгота
    :param radian_coords1:
    :param radian_coords2:
    :return:
    '''
    result = haversine_distances([radian_coords1], [radian_coords2])
    result_in_km = result[0, 0] * EARTH_SIZE
    return result_in_km


def find_buffer_width(center, target_buffer_km):
    '''
    Нам нужно подобрать такую ширину буффера, чтобы расстояние по сфере с такой шириной было не меньше ширины в км
    :param center: центр точки для расчета в окрестности в градусах(широта, долгота)
    :param target_buffer: размер целевого буфера, км
    :return: размер буфера в градусах
    '''
    center = np.asarray(center)

    def find_buffer_width_min_fn(p):
        long_dist = haver_distance(np.radians(center), np.radians(center + [0, p]))
        lat_dist = haver_distance(np.radians(center), np.radians(center + [p, 0]))

        delta = target_buffer_km - min(long_dist, lat_dist)

        return delta * delta

    result = newton_krylov(find_buffer_width_min_fn, 0)

    return result


if __name__ == '__main__':
    root = find_buffer_width([56, 30], 0.003)
    print(root)

