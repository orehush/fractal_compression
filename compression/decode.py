import numpy as np
from PIL import Image
from scipy.ndimage import affine_transform

from compression.affine_transform import (
    AFFINE_TRANSFORMS_3D, AFFINE_TRANSFORMS_2D
)
from compression.utils import (
    scale_matrix_by_size, get_transform_matrix_for_domain
)
from compression.const import (
    TRANSFORMATION, RANGE_BLOCK_SIZE, DOMAIN_BLOCK_SIZE,
)


class FractalDecode(object):

    def __init__(self, transformations=TRANSFORMATION,
                 range_size=RANGE_BLOCK_SIZE,
                 domain_size=DOMAIN_BLOCK_SIZE,
                 width=512, height=512, brightness_ratio=0.7):

        self.transformations = transformations
        self.range_size = range_size
        self.domain_size = domain_size
        self.domain_transform_matrix = get_transform_matrix_for_domain(
            self.domain_size, self.range_size
        )

        self.brightness_ratio = brightness_ratio

        self.width = width
        self.height = height

        self.img_data = np.zeros((self.width, self.height), dtype=np.int16)
        self.new_img_data = np.zeros((self.width, self.height), dtype=np.int16)

        self.iterations = 0

    def apply_transforms(self):
        for transform in self.transformations:
            self.apply_transform(transform)
        self.img_data = self.new_img_data.copy()
        self.iterations += 1

    def apply_transform(self, transform):
        rx, ry, dx, dy = tuple(map(int, transform[:4]))
        afft = AFFINE_TRANSFORMS_2D[int(transform[4])]
        color_shift = transform[5]

        domain = self.img_data[
                 dx:dx + self.domain_size, dy:dy + self.domain_size]
        # if domain.shape[:2] != (DOMAIN_BLOCK_SIZE, DOMAIN_BLOCK_SIZE):
        #     continue
        scaled_domain = np.round(
            scale_matrix_by_size(domain, self.domain_transform_matrix) *
            self.brightness_ratio - np.array(color_shift)
        )
        scaled_transformed_domain = affine_transform(
            input=scaled_domain,
            matrix=afft.get_matrix(),
            offset=afft.get_offset(scaled_domain.shape)
        )
        self.new_img_data[
            rx:rx + self.range_size, ry:ry + self.range_size
        ] = scaled_transformed_domain

    def show_current(self):
        Image.fromarray(self.img_data).show()
