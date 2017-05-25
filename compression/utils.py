from datetime import datetime
from typing import List

import numpy as np


def get_transform_matrix_for_domain(domain_size: int,
                                    range_size: int) -> np.ndarray:
    """
    Build matrix for scale domain matrix to range's size
    :param domain_size: int
    :param range_size: int
    :return: np.ndarray - array of size (domain_size, range_size)
    """
    delta = domain_size//range_size
    transform_matrix = np.zeros((domain_size, range_size), np.float16)
    for i in range(range_size):
        for j in range(delta):
            transform_matrix[delta * i + j, i] = 1.0/delta
    return transform_matrix


def scale_matrix_by_size(matrix: np.ndarray, transform_matrix: np.ndarray):
    """

    :param matrix: np.ndarray
    :param transform_matrix: np.ndarray
    :return: np.ndarray
    """
    shape = matrix.shape
    if len(shape) == 3:
        return np.transpose(np.array(
            [
                scale_matrix_by_size(matrix[:, :, i], transform_matrix)
                for i in range(3)
            ]
        ), axes=[0, 2, 1]).T
    if len(shape) == 2:
        return matrix.dot(transform_matrix).T.dot(transform_matrix).T
    raise ValueError("Scale not support shape %s " % shape)


def get_color_shift_for_blocks(range_block: np.ndarray,
                               domains: np.ndarray) -> np.ndarray:
    """

    :param range_block: np.ndarray - one range block
    :param domains: np.ndarray - array of domains blocks
                    with the same size as range_block
    :return: np.ndarray - calculate average distance
             for each domain arrays pixels and range block array pixels
    """
    width, height = range_block.shape[:2]
    return (domains - range_block).sum(axis=2).sum(axis=1) / (width * height)


def get_index_for_closest_domain_to_range(range_block: np.ndarray,
                                          domains: np.ndarray,
                                          color_shift: np.ndarray) \
        -> np.ndarray:
    """

    :param range_block: np.ndarray - one range block
    :param domains: np.ndarray - array of domains blocks
                    with the same size as range_block
    :param color_shift: np.ndarray - average distance
           for each domain arrays pixels and range block array pixels
    :return: int - index of closest domain to range
    """
    dim = len(range_block.shape)
    if dim == 3:
        return np.fabs(
            np.transpose(
                np.transpose(domains, axes=[1, 2, 0, 3]) - color_shift,
                axes=[2, 0, 1, 3]
            ) - range_block
        ).sum(axis=3).sum(axis=2).sum(axis=1).argmin()
    if dim == 2:
        return np.fabs(
            np.transpose(domains.T - color_shift.T) - range_block
        ).sum(axis=2).sum(axis=1).argmin()
    raise ValueError("Arrays with dimensions %s not supported" % dim)


def scaled_all_blocks(domains: np.ndarray,
                      scaled_transform_matrix: np.ndarray,
                      brightness_coefficient: float) -> np.ndarray:
    """

    :param domains: np.ndarray - array with domain blocks arrays
    :param scaled_transform_matrix: np.ndarray - matrix, used for transform
           each domain to size as range block
    :param brightness_coefficient: float - aka compression coefficient
    :return: np.ndarray -
    """

    def scale_matrix(domain):
        return scale_matrix_by_size(domain, scaled_transform_matrix)

    return np.array(list(map(
        scale_matrix, domains
    ))) * brightness_coefficient


def get_fractal_transformations(ranges: np.ndarray,
                                domains: np.ndarray,
                                ranges_indexes: List[tuple],
                                domains_indexes: List[tuple],
                                callback=None) -> np.ndarray:
    """

    :param ranges: np.ndarray
    :param domains: np.ndarray
    :param ranges_indexes: list of tuples
    :param domains_indexes: list of tuples
    :param callback: function
    :return:
    """
    now = datetime.now()
    fractal_transformations = np.zeros((ranges.shape[0], 6), dtype=np.uint16)

    for i, range_block in enumerate(ranges):

        color_shift = get_color_shift_for_blocks(range_block, domains)
        closest_domain_index = get_index_for_closest_domain_to_range(
            range_block, domains, color_shift
        )

        chosen_color_shift = tuple(color_shift[closest_domain_index]) \
            if isinstance(color_shift[0], np.ndarray) \
            else (color_shift[closest_domain_index], )

        fractal_transformations[i] = np.array(
            ranges_indexes[i] +
            domains_indexes[closest_domain_index] +
            chosen_color_shift
        )
        if i % 100 == 0 and callable(callback):
            callback(i)
            print(datetime.now() - now)
