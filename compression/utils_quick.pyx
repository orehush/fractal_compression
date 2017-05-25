from cython.parallel import prange
cimport cython
import numpy as np
cimport numpy as np


def get_color_shift_for_blocks(np.ndarray range_block,
                               np.ndarray domains):
    """

    :param range_block: np.ndarray - one range block
    :param domains: np.ndarray - array of domains blocks
                    with the same size as range_block
    :return: np.ndarray - calculate average distance
             for each domain arrays pixels and range block array pixels
    """
    cdef Py_ssize_t width = range_block.shape[0]
    cdef Py_ssize_t height = range_block.shape[1]
    return - (domains - range_block).sum(axis=2).sum(axis=1) / (width * height)


def get_index_for_closest_domain_to_range(np.ndarray range_block,
                                          np.ndarray domains,
                                          np.ndarray color_shift):
    """

    :param range_block: np.ndarray - one range block
    :param domains: np.ndarray - array of domains blocks
                    with the same size as range_block
    :param color_shift: np.ndarray - average distance
           for each domain arrays pixels and range block array pixels
    :return: int - index of closest domain to range
    """
    cdef Py_ssize_t shape = range_block.ndim
    if shape == 3:
        return np.fabs(
            np.transpose(
                np.transpose(domains, axes=[1, 2, 0, 3]) + color_shift,
                axes=[2, 0, 1, 3]
            ) - range_block
        ).sum(axis=3).sum(axis=2).sum(axis=1).argmin()
    if shape == 2:
        return np.fabs(
            np.transpose(domains.T + color_shift.T) - range_block
        ).sum(axis=2).sum(axis=1).argmin()
    raise ValueError("Arrays with dimensions %s not supported" % len(shape))


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_fractal_transformations(np.ndarray ranges,
                                np.ndarray domains,
                                list ranges_indexes,
                                list domains_indexes):
    """

    :param ranges: np.ndarray
    :param domains: np.ndarray
    :param ranges_indexes: list of tuples
    :param domains_indexes: list of tuples
    :return:
    """
    # now = datetime.now()
    cdef Py_ssize_t n = ranges.shape[0]
    cdef np.ndarray fractal_transformations = np.zeros((n, 6))
    cdef Py_ssize_t i
    for i in prange(n, nogil=True):

        color_shift = get_color_shift_for_blocks(ranges[i], domains)
        closest_domain_index = get_index_for_closest_domain_to_range(
            ranges[i], domains, color_shift
        )

        with gil:
            chosen_color_shift = tuple(color_shift[closest_domain_index]) \
                if isinstance(color_shift[0], np.ndarray) \
                else (color_shift[closest_domain_index], )

            fractal_transformations[i][0] = ranges_indexes[i][0]
            fractal_transformations[i][1] = ranges_indexes[i][1]
            print(i)
            # ranges_indexes[i] +
            # domains_indexes[closest_domain_index] +
            # chosen_color_shift
        # )
        # print(i)
    return fractal_transformations
