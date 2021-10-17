import numpy as np
from math import radians



# https://blogs.sas.com/content/iml/2016/08/29/weighted-percentiles.html
# https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
# https://github.com/numpy/numpy/blob/v1.17.0/numpy/lib/function_base.py#L3569-L3713


def _weighted_percentile_1d(a, q, interpolation, w=None):
    """ Предполагается, что q лежит в  [0, 1], и a,w -  ndarray"""

    idx = np.argsort(a)
    a_sort = a[idx]
    w_sort = w[idx]

    # кумулятивная сумма
    ecdf = np.cumsum(w_sort)

    # Перцентиль
    p = q * (w.sum() - 1)

    if interpolation == 'linear':
        p = p
    if interpolation == 'lower':
        p = np.floor(p)
    if interpolation == 'higher':
        p = np.ceil(p)
    if interpolation == 'midpoint':
        p = (np.floor(p) + np.ceil(p)) * 0.5
    if interpolation == 'nearest':
        p = np.around(p)

    idx_low = np.searchsorted(ecdf, p, side='right')
    idx_high = np.searchsorted(ecdf, p + 1, side='right')
    idx_high[idx_high > ecdf.size - 1] = ecdf.size - 1

    # Подсчет весов
    weights_high = p - np.floor(p)
    weights_low = 1.0 - weights_high

    # Подсчет макс/мин индексов и умножение на соответствующие веса
    x1 = np.take(a_sort, idx_low) * weights_low
    x2 = np.take(a_sort, idx_high) * weights_high

    return np.add(x1, x2)


def weighted_percentile(a, q, interpolation='linear', w=None, axis=None):
    """
    Weighted quantile of an array with respect to the last axis.
   Parameters
    ----------
    a : array-like
        The input array from which to calculate percents
    q : float/array-like
        The percentiles to calculate (0.0 - 100.0)
    w : array-like, optional
        The weights to assign to values of a.  Equal weighting if None
        is specified
    interpolation: {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
        This optional parameter specifies the interpolation method to
        use when the desired percentile lies between two data points
        ``i < j``:
        * 'linear': ``i + (j - i) * fraction``, where ``fraction``
          is the fractional part of the index surrounded by ``i``
          and ``j``.
        * 'lower': ``i``.
        * 'higher': ``j``.
        * 'nearest': ``i`` or ``j``, whichever is nearest.
        * 'midpoint': ``(i + j) / 2``.
    axis: {int, tuple of int, None}, optional
        Axis = None is flattened list
        Axis or axes along which the percentiles are computed. The
        default is to compute the percentile(s) along a flattened
        version of the array.

    Returns
    -------
    wpercentile : np.array
        The output value.
    """
    a = np.asarray(a)

    if w is None:
        w = np.ones_like(a)
    else:
        w = np.asarray(w)

    if not np.isscalar(q):
        q = np.asarray(q)

    if np.any(q < 0) or np.any(q > 100):
        raise ValueError('percentile must be between 0 and 100')

    q = q / 100

    if isinstance(q, float):
        q = np.array([q])

    if axis is None:
        a = np.ravel(a)
        w = np.ravel(w)

    if axis == 0:
        a = np.transpose(a)
        w = np.transpose(w)

    nd = a.ndim

    if nd == 0:
        TypeError('data must have at least one dimension')

    elif nd == 1:
        return _weighted_percentile_1d(a, q, interpolation, w)

    elif nd > 1:
        n = a.shape
        imr = a.reshape((np.prod(n[:-1]), n[-1]))
        result = np.array([])
        for row, wei in zip(imr, w):
            result = np.concatenate((result, _weighted_percentile_1d(row, q, interpolation, wei)))

        return np.asarray(result)


