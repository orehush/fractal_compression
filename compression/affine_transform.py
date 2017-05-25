import numpy as np
from scipy.ndimage.interpolation import affine_transform


class AffineTransformation(object):
    matrix = []
    offset = []

    def __init__(self, matrix, offset):
        self.matrix = matrix
        self.offset = offset

    def get_matrix(self):
        return self.matrix

    def get_offset(self, dim=None):
        if dim is None:
            dim = [1]*len(self.offset)
        return list(map(lambda x, y: x*y, self.offset, dim))


# use for gray images (8-bit)
AFFINE_TRANSFORMS_2D = (
    AffineTransformation([[1, 0], [0, 1]], [0, 0]),
    AffineTransformation([[0, 1], [-1, 0]], [0, 1]),
    AffineTransformation([[-1, 0], [0, -1]], [1, 1]),
    AffineTransformation([[0, -1], [1, 0]], [1, 0]),
    AffineTransformation([[-1, 0], [0, 1]], [1, 0]),
    AffineTransformation([[1, 0], [0, -1]], [0, 1]),
    AffineTransformation([[0, -1], [0, 1]], [1, 0]),
    AffineTransformation([[0, 1], [1, 0]], [0, 0]),
)

# use for color images (24-bit)
AFFINE_TRANSFORMS_3D = (
    AffineTransformation([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0]),
    AffineTransformation([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], [0, 1, 0]),
    AffineTransformation([[0, -1, 0], [-1, 0, 0], [0, 0, 1]], [1, 1, 0]),
    AffineTransformation([[0, -1, 0], [1, 0, 0], [0, 0, 1]], [1, 0, 0]),
    AffineTransformation([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], [1, 0, 0]),
    AffineTransformation([[1, 0, 0], [0, -1, 0], [0, 0, 1]], [0, 1, 0]),
    AffineTransformation([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], [1, 1, 0]),
    AffineTransformation([[0, 1, 0], [1, 0, 0], [0, 0, 1]], [0, 0, 0]),
)


def affine_transform_coord(x: int, y: int,
                           affine_matrix_transform: AffineTransformation,
                           dim=None) -> tuple:
    return tuple(
        np.dot(
            np.array(affine_matrix_transform.get_matrix())[0:2, 0:2],
            np.array([x, y])
        ) + np.array(affine_matrix_transform.get_offset(dim))[0:2]
    )


def affine_transform_generator(img_data: np.array):
    """

    :param img_data: np.array
    :return: GeneratorType:
    """
    shape = img_data.shape
    if len(shape) == 3:
        transforms = AFFINE_TRANSFORMS_3D
    elif len(shape) == 2:
        transforms = AFFINE_TRANSFORMS_2D
    else:
        raise ValueError("Not supported shape %s. Affine transforms support "
                         "only 2 or 3 dimensions arrays" % shape)

    for i, transform in enumerate(transforms):
        yield affine_transform(
            input=img_data,
            matrix=transform.get_matrix(),
            offset=transform.get_offset(shape)
        ), i
