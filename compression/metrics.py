import numpy as np


def measure(range_block, domain_block, color_shift):
    """

    :param range_block: np.array
    :param domain_block: np.array
    :param color_shift: float
    :return: float
    """
    return (
        (range_block - (domain_block + color_shift))**2
    ).sum()


def scale_matrix_by_size(matrix, transform_matrix):
    """

    :param matrix: np.array
    :param transform_matrix: np.array
    :return: np.array
    """
    return matrix.dot(transform_matrix).T.dot(transform_matrix).T


def merge_with_ones_vector(a):
    """
    :param a - np.array
    :return np.array
    """
    ones_vector = np.ones(a.shape, dtype=np.int16)
    return np.concatenate((ones_vector, a), axis=1)


def reshape_to_vector(a):
    """
    :param a - np.array
    :return np.array
    """
    return np.array(a.reshape((a.shape[0]**2, 1)), dtype=np.int16)


def distance_L2(a, b):
    """
    :param a - np.array
    :param b - np.array
    :return float
    """
    return ((a-b)**2).sum()


def ols(a, b):
    """
    :param a - np.array - square block
    :param b - np.array - square block
    :return np.array, float - coeficients of ols, error
    """
    a = reshape_to_vector(a)
    b = reshape_to_vector(b)
    A = merge_with_ones_vector(a)
    beta = np.dot(np.linalg.inv(A.T.dot(A)), A.T.dot(b))
    return beta, distance_L2(A.dot(beta), b)


def calculate_mse(first_array, second_array):
    """

    :param first_array: np.array
    :param second_array: np.array
    :return:
    """
    return ((first_array - second_array)**2).sum()/\
           (first_array.shape[0]*first_array.shape[1])


def calculate_psnr(first_array, second_array):
    """

    :param first_array: np.array
    :param second_array: np.array
    :return:
    """

    return 10*np.log10((255**2)/calculate_mse(first_array, second_array))
