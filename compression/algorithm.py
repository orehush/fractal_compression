from datetime import datetime

import numpy as np

from PIL import Image

from compression.affine_transform import affine_transform_generator
from compression.utils import scaled_all_blocks, \
    get_transform_matrix_for_domain, get_fractal_transformations

RANGE_BLOCK_SIZE = 8
DOMAIN_BLOCK_SIZE = 16
BRIGHTNESS_COEFFICIENT = 0.7


class FractalCompressor(object):

    def __init__(self, range_size: int, domain_size: int,
                 brightness_coefficient: float):
        now = datetime.now()
        self.range_size = range_size
        self.domain_size = domain_size
        self.brightness_coefficient = brightness_coefficient
        self.domain_transform_matrix = get_transform_matrix_for_domain(
            self.domain_size, self.range_size
        )
        self.range_indexes = []
        self.domain_indexes = []
        print("Initializing: ", datetime.now() - now)

    def split_into_range_blocks(self, img_data: np.ndarray) -> np.ndarray:
        """
        Split array of image data to range blocks. Saving indexes for each range
        :param img_data: np.ndarray
        :return: np.ndarray - array of ranges blocks
        """
        rows, cols = img_data.shape[:2]
        blocks = []
        size = self.range_size
        for row in range(0, rows, size):
            for col in range(0, cols, size):
                blocks.append(img_data[row: row + size, col: col + size])
                self.range_indexes.append((row, col))
        return np.array(blocks)

    def split_into_domain_blocks(self, img_data: np.ndarray) -> np.ndarray:
        """
        Split array of image data to domain blocks.
        Saving indexes for each domain
        :param img_data: np.ndarray
        :return: np.ndarray - array of domains blocks
        """

        rows, cols = img_data.shape[:2]
        blocks = []
        size = self.domain_size
        shift = self.domain_size // 2
        for row in range(0, rows, shift):
            for col in range(0, cols, shift):
                domain = img_data[row: row + size, col: col + size]
                if domain.shape[:2] != (size, size):
                    continue
                for transformed_domain, number in affine_transform_generator(
                        domain):
                    blocks.append(transformed_domain)
                    self.domain_indexes.append((row, col, number, ))
        return np.array(blocks)

    @staticmethod
    def _get_image_data(image):
        if isinstance(image, Image.Image):
            return np.array(image)
        if isinstance(image, str):
            return np.array(Image.open(image))
        if isinstance(image, np.ndarray):
            return image
        raise ValueError("Not support type: %s" % type(image))

    def compress(self, image):
        img_data = self._get_image_data(image)

        now = datetime.now()
        ranges = self.split_into_range_blocks(img_data)
        print("Split in ranges: ", datetime.now() - now)

        now = datetime.now()
        domains = self.split_into_domain_blocks(img_data)
        print("Split in domains: ", datetime.now() - now)

        now = datetime.now()
        scaled_domains = scaled_all_blocks(
            domains, self.domain_transform_matrix, self.brightness_coefficient
        )
        print("Scaled domains: ", datetime.now() - now)

        transformation = get_fractal_transformations(
            ranges, scaled_domains, self.range_indexes, self.domain_indexes,
            callback=lambda i: print("Iteration %s" % i)
        )
        return transformation


if __name__ == '__main__':
    compressor = FractalCompressor(RANGE_BLOCK_SIZE, DOMAIN_BLOCK_SIZE,
                                   BRIGHTNESS_COEFFICIENT)
    print(compressor.compress('../img/einstein.bmp'))
