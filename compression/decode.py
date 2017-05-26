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
        d_x0, d_y0 = int(t[2]), int(t[3])
        transform1 = AFFINE_TRANSFORMS_2D[int(t[4])]
        d_x1, d_y1 = int(t[5]), int(t[6])
        transform2 = AFFINE_TRANSFORMS_2D[int(t[7])]
        d_x2, d_y2 = int(t[8]), int(t[9])
        transform3 = AFFINE_TRANSFORMS_2D[int(t[10])]
        color_shift0 = t[11]
        color_shift1 = t[12]
        color_shift2 = t[13]

        domain = np.zeros((DOMAIN_BLOCK_SIZE, DOMAIN_BLOCK_SIZE, 3), dtype=np.float32)
        domain[:, :, 0] = img_data[d_x0:d_x0+DOMAIN_BLOCK_SIZE, d_y0:d_y0+DOMAIN_BLOCK_SIZE, 0]
        domain[:, :, 1] = img_data[d_x1:d_x1+DOMAIN_BLOCK_SIZE, d_y1:d_y1+DOMAIN_BLOCK_SIZE, 1]
        domain[:, :, 2] = img_data[d_x2:d_x2+DOMAIN_BLOCK_SIZE, d_y2:d_y2+DOMAIN_BLOCK_SIZE, 2]
        # if domain.shape[:2] != (DOMAIN_BLOCK_SIZE, DOMAIN_BLOCK_SIZE):
        #     continue
        scaled_domain = np.zeros((RANGE_BLOCK_SIZE, RANGE_BLOCK_SIZE, 3), dtype=np.float32)
        scaled_domain[:, :, 0] = np.round(scale_matrix_by_size(domain[:, :, 0], domain_transform_matrix) * 0.7 - np.array(color_shift0))
        scaled_domain[:, :, 1] = np.round(scale_matrix_by_size(domain[:, :, 1], domain_transform_matrix) * 0.7 - np.array(color_shift1))
        scaled_domain[:, :, 2] = np.round(scale_matrix_by_size(domain[:, :, 2], domain_transform_matrix) * 0.7 - np.array(color_shift2))

        scaled_transformed_domain = np.zeros((RANGE_BLOCK_SIZE, RANGE_BLOCK_SIZE, 3), dtype=np.float32)
        scaled_transformed_domain[:, :, 0] = affine_transform(
            input=scaled_domain[:, :, 0],
            matrix=transform1.get_matrix(),
            offset=transform1.get_offset(scaled_domain.shape[:2])
        )
        scaled_transformed_domain[:, :, 1] = affine_transform(
            input=scaled_domain[:, :, 1],
            matrix=transform1.get_matrix(),
            offset=transform1.get_offset(scaled_domain.shape[:2])
        )
        scaled_transformed_domain[:, :, 2] = affine_transform(
            input=scaled_domain[:, :, 2],
            matrix=transform2.get_matrix(),
            offset=transform2.get_offset(scaled_domain.shape[:2])
        )
        new_img_data[r_x:r_x+RANGE_BLOCK_SIZE, r_y:r_y+RANGE_BLOCK_SIZE] = scaled_transformed_domain
    img_data = new_img_data.copy()
    print(img_data.shape)
    print(img_data.dtype)

    Image.fromarray(img_data, 'RGB').show()
    input()
