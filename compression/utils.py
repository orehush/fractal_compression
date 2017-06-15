from datetime import datetime
from typing import List

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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


def clustering_blocks(ranges, domains, scaler='minmax'):
    """

    :param ranges:
    :param domains:
    :param scaler: 'minmax' or 'standard'
    :return:
    """
    all_blocks = np.concatenate((ranges, domains))
    means = all_blocks.mean(axis=(1, 2), dtype=np.float32)
    stds = all_blocks.std(axis=(1, 2), dtype=np.float32)
    points = np.matrix([means, stds]).T

    if scaler == 'minmax':
        points = MinMaxScaler().fit_transform(points)
    elif scaler == 'standard':
        points = StandardScaler().fit_transform(points)
    else:
        raise ValueError("Not support scaler type %s " % scaler)

    labels = KMeans(n_clusters=30).fit_predict(points)
    labels_set = set(labels)
    range_labels = labels[0:ranges.shape[0]]
    domain_labels = labels[ranges.shape[0]:]
    return labels_set, range_labels, domain_labels


def get_fractal_transformations(ranges: np.ndarray,
                                domains: np.ndarray,
                                ranges_indexes: List[tuple],
                                domains_indexes: List[tuple],
                                divide_into_clusters=False,
                                callback=None) -> list:
    """

    :param ranges: np.ndarray
    :param domains: np.ndarray
    :param ranges_indexes: list of tuples
    :param domains_indexes: list of tuples
    :param callback: function
    :return:
    """

    if not divide_into_clusters:
        return search_transformations(
            ranges, domains, ranges_indexes, domains_indexes
        )

    fractal_transformations = []
    ranges_indexes = np.array(ranges_indexes)
    domains_indexes = np.array(domains_indexes)
    labels_set, range_labels, domain_labels = clustering_blocks(ranges, domains)

    for group in labels_set:
        ranges_conditions = range_labels == group
        domains_conditions = domain_labels == group

        group_ranges = ranges[ranges_conditions]
        group_domains = domains[domains_conditions]
        group_ranges_indexes = ranges_indexes[ranges_conditions]
        group_domains_indexes = domains_indexes[domains_conditions]
        if group_ranges.shape[0] == 0:
            continue

        if group_domains.shape[0] * 2 < group_ranges.shape[0]:
            group_domains = domains
            group_domains_indexes = domains_indexes

        fractal_transformations.extend(
            search_transformations(
                group_ranges,
                group_domains,
                group_ranges_indexes,
                group_domains_indexes
            )
        )

    return fractal_transformations


def search_transformations(ranges, domains, ranges_indexes, domains_indexes):
    now = datetime.now()
    fractal_transformations = []
    for i, range_block in enumerate(ranges):
        delta = domains - range_block

        # calculate average distance for each domain arrays pixels
        # and range block array pixels
        color_shift = np.mean(delta, axis=(2, 1))

        closest_domain_index = get_index_for_closest_domain_to_range(
            delta, color_shift
        )

        if isinstance(color_shift[0], np.ndarray):
            chosen_color_shift = tuple(color_shift[closest_domain_index].astype(int))
        else:
            chosen_color_shift = (np.int(color_shift[closest_domain_index]), )

        fractal_transformations.append((
            tuple(ranges_indexes[i]) +
            tuple(domains_indexes[closest_domain_index]) +
            chosen_color_shift
        ))
        if i % 100 == 0:
            print(datetime.now() - now)
            now = datetime.now()
    return fractal_transformations
