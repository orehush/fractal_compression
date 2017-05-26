import numpy as np
from PIL import Image
from scipy.ndimage import affine_transform

from compression.affine_transform import AFFINE_TRANSFORMS_3D, \
    AFFINE_TRANSFORMS_2D
from compression.utils import scale_matrix_by_size, \
    get_transform_matrix_for_domain
from compression.const import TRANSFORMATION

RANGE_BLOCK_SIZE = 8
DOMAIN_BLOCK_SIZE = 16

img_data = np.zeros((512, 512, 3), dtype=np.int16)

domain_transform_matrix = get_transform_matrix_for_domain(DOMAIN_BLOCK_SIZE, RANGE_BLOCK_SIZE)

for _ in range(15):
    new_img_data = np.zeros((512, 512, 3), dtype=np.int16)
    for t in TRANSFORMATION:
        r_x, r_y = int(t[0]), int(t[1])
        d_x, d_y = int(t[2]), int(t[3])
        transform = AFFINE_TRANSFORMS_3D[int(t[4])]
        color_shift = t[5:]



        domain = img_data[d_x:d_x+DOMAIN_BLOCK_SIZE, d_y:d_y+DOMAIN_BLOCK_SIZE]
        # if domain.shape[:2] != (DOMAIN_BLOCK_SIZE, DOMAIN_BLOCK_SIZE):
        #     continue
        scaled_domain = np.round(scale_matrix_by_size(domain, domain_transform_matrix) * 0.7 - np.array(color_shift))
        scaled_transformed_domain = affine_transform(
            input=scaled_domain,
            matrix=transform.get_matrix(),
            offset=transform.get_offset(scaled_domain.shape)
        )
        new_img_data[r_x:r_x+RANGE_BLOCK_SIZE, r_y:r_y+RANGE_BLOCK_SIZE] = scaled_transformed_domain
    img_data = new_img_data.copy()
    print(img_data.shape)
    print(img_data.dtype)

    Image.fromarray(img_data, 'RGB').show()
    input()
