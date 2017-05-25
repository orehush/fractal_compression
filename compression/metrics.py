import numpy as np


def scale_matrix_by_size(matrix: np.ndarray, transform_matrix: np.ndarray):
    """

    :param matrix: np.ndarray
    :param transform_matrix: np.ndarray
    :return: np.ndarray
    """
    return matrix.dot(transform_matrix).T.dot(transform_matrix).T


def merge_with_ones_vector(a: np.ndarray) -> np.ndarray:
    """
    :param a - np.ndarray
    :return np.ndarray
    """
    ones_vector = np.ones(a.shape, dtype=np.int16)
    return np.concatenate((ones_vector, a), axis=1)


def reshape_to_vector(a):
    """
    :param a - np.ndarray
    :return np.ndarray
    """
    return np.array(a.reshape((a.shape[0]**2, 1)), dtype=np.int16)


def distance_L2(a, b):
    """
    :param a - np.ndarray
    :param b - np.ndarray
    :return float
    """
    return ((a-b)**2).sum()


def ols(a, b):
    """
    :param a - np.ndarray - square block
    :param b - np.ndarray - square block
    :return np.ndarray, float - coeficients of ols, error
    """
    a = reshape_to_vector(a)
    b = reshape_to_vector(b)
    A = merge_with_ones_vector(a)
    beta = np.dot(np.linalg.inv(A.T.dot(A)), A.T.dot(b))
    return beta, distance_L2(A.dot(beta), b)


def calculate_mse(first_array: np.ndarray, second_array: np.ndarray) -> float:
    """

    :param first_array: np.ndarray
    :param second_array: np.ndarray
    :return: float
    """
    width, height = first_array.shape[:2]
    return ((first_array - second_array)**2).sum() / (width * height)


def calculate_psnr(first_array: np.ndarray, second_array: np.ndarray) -> float:
    """

    :param first_array: np.ndarray
    :param second_array: np.ndarray
    :return: float
    """

    return 10*np.log10((255**2)/calculate_mse(first_array, second_array))
