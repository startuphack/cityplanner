import numpy as np
import pickle
import cloudpickle
from sklearn.metrics.pairwise import haversine_distances
from sklearn.neighbors import BallTree
import pygeos
import itertools as it
from shapely.strtree import STRtree


def test_ball_tree_index():
    x = np.radians(np.array(
        [
            [55.7765025191759, 37.74713353831271],
            [56.7765025191759, 38.74713353831271],
            [57.7765025191759, 39.74713353831271],
        ]
    ))
    target = np.radians(np.array([[55.77353305625259, 37.748014376195776]]))

    tree = BallTree(x, metric='haversine')

    distances, indices = tree.query(target)

    assert distances[0, 0] == haversine_distances(x, target)[0, 0]


def test_pygeos_tree_index():
    tree = pygeos.STRtree(pygeos.points(np.arange(10), np.arange(10)))
    bytes = cloudpickle.dumps(tree)
    tree = cloudpickle.loads(bytes)
    # Query geometries that overlap envelope of input geometries:
    assert tree.query(pygeos.box(2, 2, 4, 4)).tolist() == [2, 3, 4]
    # Query geometries that are contained by input geometry:
    assert tree.query(pygeos.box(2, 2, 4, 4), predicate='contains').tolist() == [3]
    # Query geometries that overlap envelopes of ``geoms``
    assert tree.query_bulk([pygeos.box(2, 2, 4, 4), pygeos.box(5, 5, 6, 6)]).tolist() == [[0, 0, 0, 1, 1],
                                                                                          [2, 3, 4, 5, 6]]
    assert tree.nearest([pygeos.points(1, 1), pygeos.points(3, 5)]).tolist() == [[0, 1], [1, 4]]




def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def test_quad_tree_index():
    np.random.seed(0)
    boxes = list()
    for ix, iy in it.product(np.arange(10), np.arange(10)):
        boxes.append(pygeos.box(ix, iy, ix + 1, iy + 1))

    points = pygeos.points(np.arange(10) + 0.5, np.arange(10) + 0.5)

    tree = pygeos.STRtree(boxes)

    p_indices, box_indices = tree.query_bulk(points).tolist()
    for p_idx, box_idx in zip(p_indices, box_indices):
        point = points[p_idx]
        box = boxes[box_idx]
        assert pygeos.contains(box, point)

        print(f'{box} includes {point}')
