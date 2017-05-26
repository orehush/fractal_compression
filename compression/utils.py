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


def get_index_for_closest_domain_to_range(delta: np.ndarray,
                                          color_shift: np.ndarray) \
        -> np.ndarray:
    """

    :param delta: np.ndarray - delta before domains blocks and range block
    :param color_shift: np.ndarray - average distance
           for each domain arrays pixels and range block array pixels
    :return: int - index of closest domain to range
    """
    dim = len(delta.shape)
    if dim == 4:
        return np.sum(np.fabs(
            np.transpose(delta, axes=[1, 2, 0, 3]) - color_shift,
        ), axis=(0, 1, 3)).argmin()
    if dim == 3:
        return np.sum(np.fabs(delta.T - color_shift.T), axis=(1, 0)).argmin()
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

    return np.array(list(map(scale_matrix, domains))) * brightness_coefficient


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
    transformations_shape = ranges.shape[0], 14
    fractal_transformations = np.zeros(transformations_shape, dtype=np.float32)

    for i, range_block in enumerate(ranges):
        delta = domains - range_block

        # calculate average distance for each domain arrays pixels
        # and range block array pixels
        color_shift = np.mean(delta, axis=(2, 1))

        closest_domain_index0 = get_index_for_closest_domain_to_range(
            delta[:, :, :, 0], color_shift[:, 0]
        )
        closest_domain_index1 = get_index_for_closest_domain_to_range(
            delta[:, :, :, 1], color_shift[:, 1]
        )
        closest_domain_index2 = get_index_for_closest_domain_to_range(
            delta[:, :, :, 2], color_shift[:, 2]
        )
        chosen_color_shift0 = color_shift[closest_domain_index0, 0]
        chosen_color_shift1 = color_shift[closest_domain_index1, 1]
        chosen_color_shift2 = color_shift[closest_domain_index2, 2]

        # chosen_color_shift = tuple(color_shift[closest_domain_index]) \
        #     if isinstance(color_shift[0], np.ndarray) \
        #     else (color_shift[closest_domain_index], )

        fractal_transformations[i] = np.array(
            ranges_indexes[i] +
            domains_indexes[closest_domain_index0] +
            domains_indexes[closest_domain_index1] +
            domains_indexes[closest_domain_index2] +
            (chosen_color_shift0, chosen_color_shift1, chosen_color_shift2),
            dtype=np.float32
        )
        if i % 100 == 0 and callable(callback):
            callback(i)
            print(datetime.now() - now)
    return fractal_transformations
