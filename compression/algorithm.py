from datetime import datetime
from random import randint
import numpy as np

from PIL import Image

from compression.affine_transform import affine_transform_generator
from compression.utils import (
    scaled_all_blocks,
    get_transform_matrix_for_domain,
    search_transformations_within_clusters,
    search_transformations,
    get_deltas)
from compression.const import (
    RANGE_BLOCK_SIZE,
    DOMAIN_BLOCK_SIZE,
    BRIGHTNESS_COEFFICIENT
)


class FractalCompressor(object):

    def __init__(self, range_size: int, domain_size: int,
                 brightness_coefficient: float):
        self.range_size = range_size
        self.domain_size = domain_size

        self.brightness_coefficient = brightness_coefficient

        self.domain_transform_matrix = get_transform_matrix_for_domain(
            self.domain_size, self.range_size
        )

        self.range_indexes = []
        self.domain_indexes = []

        # those variables will be updated after load image
        self.img_data = None
        self.ranges = None
        self.domains = None
        self.scaled_domains = None

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

    def split_into_random_domain_blocks(self, img_data: np.ndarray,
                                        amount=4000) -> np.ndarray:
        """
        Split image into random domains
        :param img_data:
        :param amount:
        :return:
        """
        rows, cols = img_data.shape[:2]
        size = self.domain_size
        points_set = set()
        while len(points_set) <= amount:
            points_set.add(
                (
                    randint(0, rows - size),
                    randint(0, cols - size),
                )
            )
        blocks = []
        for x, y in points_set:
            domain = img_data[x: x+size, y:y+size]
            if domain.shape[:2] != (size, size):
                continue
            for transformed_domain, number in affine_transform_generator(
                    domain):
                blocks.append(transformed_domain)
                self.domain_indexes.append((x, y, number,))

        return np.array(blocks)

    @staticmethod
    def _get_image_data(image):
        if isinstance(image, Image.Image):
            return np.array(image)
        if isinstance(image, str):
            return np.array(Image.open(image))
        if isinstance(image, np.ndarray):
            return image
        raise TypeError("Not support type: %s" % type(image))

    def load(self, image, use_random_domains=False):
        now = datetime.now()
        self.img_data = self._get_image_data(image)
        print('Loaded image', datetime.now() - now)

        now = datetime.now()
        self.ranges = self.split_into_range_blocks(self.img_data)
        print('Divided into range blocks', datetime.now() - now)

        now = datetime.now()
        if use_random_domains:
            self.domains = self.split_into_random_domain_blocks(self.img_data)
        else:
            self.domains = self.split_into_domain_blocks(self.img_data)
        print('Divided into domains blocks', datetime.now() - now)

        now = datetime.now()
        self.scaled_domains = scaled_all_blocks(
            self.domains,
            self.domain_transform_matrix,
            self.brightness_coefficient
        )
        print('Scaled domain blocks', datetime.now() - now)

        print('Range blocks:', len(self.ranges))
        print('Domain blocks:', len(self.domains))

    @staticmethod
    def get_min_difference(domain_deltas):
        return np.min(np.abs(np.mean(domain_deltas, axis=(1, 2))))

    def coefficient_similarity(self, func='max'):
        min_differences = list(map(
            self.get_min_difference,
            get_deltas(self.scaled_domains, self.ranges)
        ))

        if func == 'max':
            return np.max(min_differences)
        elif func == 'avg':
            return np.average(min_differences)

        raise ValueError(
            "Not supported function %s for coeficient similarity" % func
        )

    def compress(self, quick_algorithm=True):
        """
        :param quick_algorithm: bool - use quick algorithm with clustering blocks
        :return:
        """

        if quick_algorithm:
            transformation = search_transformations_within_clusters(
                self.ranges, self.scaled_domains,
                self.range_indexes, self.domain_indexes,
            )
        else:
            transformation = search_transformations(
                get_deltas(self.scaled_domains, self.ranges),
                self.range_indexes, self.domain_indexes
            )
        return transformation


if __name__ == '__main__':
    compressor = FractalCompressor(RANGE_BLOCK_SIZE, DOMAIN_BLOCK_SIZE,
                                   BRIGHTNESS_COEFFICIENT)
    print('mountain2-8.bmp')
    now = datetime.now()
    compressor.load('../img/samples/mountain2-8.bmp')
    print('Quick algorithm')
    print(compressor.compress(quick_algorithm=True))
    print(datetime.now() - now)
    print('Standard algorithm')
    print(compressor.compress(quick_algorithm=False))
    print(datetime.now() - now)

    print('mountain1-8.bmp')
    now = datetime.now()
    compressor.load('../img/samples/mountain1-8.bmp')
    print('Quick algorithm')
    print(compressor.compress(quick_algorithm=True))
    print(datetime.now() - now)
    print('Standard algorithm')
    print(compressor.compress(quick_algorithm=False))
    print(datetime.now() - now)

    print('hill-8.bmp')
    now = datetime.now()
    compressor.load('../img/samples/hill-8.bmp')
    print('Quick algorithm')
    print(compressor.compress(quick_algorithm=True))
    print(datetime.now() - now)
    print('Standard algorithm')
    print(compressor.compress(quick_algorithm=False))
    print(datetime.now() - now)

    print('sea1-8.bmp')
    now = datetime.now()
    compressor.load('../img/samples/sea1-8.bmp')
    print('Quick algorithm')
    print(compressor.compress(quick_algorithm=True))
    print(datetime.now() - now)
    print('Standard algorithm')
    print(compressor.compress(quick_algorithm=False))
    print(datetime.now() - now)

    print('sea2-8.bmp')
    now = datetime.now()
    compressor.load('../img/samples/sea2-8.bmp')
    print('Quick algorithm')
    print(compressor.compress(quick_algorithm=True))
    print(datetime.now() - now)
    print('Standard algorithm')
    print(compressor.compress(quick_algorithm=False))
    print(datetime.now() - now)

    print('sea-beach-8.bmp')
    now = datetime.now()
    compressor.load('../img/samples/sea-beach-8.bmp')
    print('Quick algorithm')
    print(compressor.compress(quick_algorithm=True))
    print(datetime.now() - now)
    print('Standard algorithm')
    print(compressor.compress(quick_algorithm=False))
    print(datetime.now() - now)

    print('-------------------------- RANDOM BLOCKS --------------------------'
          '\n\n\n\n\n\n\n\n\n\n')

    print('mountain2-8.bmp')
    now = datetime.now()
    compressor.load('../img/samples/mountain2-8.bmp', True)
    print('Quick algorithm')
    print(compressor.compress(quick_algorithm=True))
    print(datetime.now() - now)
    print('Standard algorithm')
    print(compressor.compress(quick_algorithm=False))
    print(datetime.now() - now)

    print('mountain1-8.bmp')
    now = datetime.now()
    compressor.load('../img/samples/mountain1-8.bmp', True)
    print('Quick algorithm')
    print(compressor.compress(quick_algorithm=True))
    print(datetime.now() - now)
    print('Standard algorithm')
    print(compressor.compress(quick_algorithm=False))
    print(datetime.now() - now)

    print('hill-8.bmp')
    now = datetime.now()
    compressor.load('../img/samples/hill-8.bmp', True)
    print('Quick algorithm')
    print(compressor.compress(quick_algorithm=True))
    print(datetime.now() - now)
    print('Standard algorithm')
    print(compressor.compress(quick_algorithm=False))
    print(datetime.now() - now)

    print('sea1-8.bmp')
    now = datetime.now()
    compressor.load('../img/samples/sea1-8.bmp', True)
    print('Quick algorithm')
    print(compressor.compress(quick_algorithm=True))
    print(datetime.now() - now)
    print('Standard algorithm')
    print(compressor.compress(quick_algorithm=False))
    print(datetime.now() - now)

    print('sea2-8.bmp')
    now = datetime.now()
    compressor.load('../img/samples/sea2-8.bmp', True)
    print('Quick algorithm')
    print(compressor.compress(quick_algorithm=True))
    print(datetime.now() - now)
    print('Standard algorithm')
    print(compressor.compress(quick_algorithm=False))
    print(datetime.now() - now)

    print('sea-beach-8.bmp')
    now = datetime.now()
    compressor.load('../img/samples/sea-beach-8.bmp', True)
    print('Quick algorithm')
    print(compressor.compress(quick_algorithm=True))
    print(datetime.now() - now)
    print('Standard algorithm')
    print(compressor.compress(quick_algorithm=False))
    print(datetime.now() - now)
