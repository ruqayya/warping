from skimage.io import imread
import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import itertools
from numpy.linalg import inv
import cv2

def get_patch_dimensions(canvas_size, transform):
    N = canvas_size[0]
    x = [np.linspace(1, canvas_size[0], N, endpoint=True), np.ones(canvas_size[1]) * canvas_size[0],
         np.linspace(1, canvas_size[0], N, endpoint=True), np.ones(canvas_size[1])]
    x = np.array(list(itertools.chain.from_iterable(x))) - 1

    N = canvas_size[1]
    y = [np.ones(canvas_size[0]), np.linspace(1, canvas_size[1], N, endpoint=True),
         np.ones(canvas_size[0]) * canvas_size[1], np.linspace(1, canvas_size[1], N, endpoint=True)]
    y = np.array(list(itertools.chain.from_iterable(y))) - 1

    points = np.array([x, y]).transpose()
    transformed_points = transform_points(points, transform)

    width = int(np.max([np.max(transformed_points[:, 0]) + 1, canvas_size[1]]))
    height = int(np.max([np.max(transformed_points[:, 1]) + 1, canvas_size[0]]))
    return [height, width]


def transform_points(points, matrix):
	pts_pad = np.hstack([points, np.ones((points.shape[0], 1))])
	points_warp = np.dot(pts_pad, matrix.T)
	return points_warp[:, :-1]


def apply_transform(image, transform):
    inv_transform = inv(transform)
    new_image_shape = get_patch_dimensions([image.shape[1], image.shape[0]], inv_transform)
    new_image = np.zeros(new_image_shape)
    for iRow in range(image.shape[0]):
        for iCol in range(image.shape[1]):
            coord = np.array([[iCol, iRow]])
            transform_coord = transform_points(coord, inv_transform)
            if (not (transform_coord < 0).any()) and (transform_coord[0, 0] < image.shape[1]) and (transform_coord[0, 1] < image.shape[0]):
                new_image[iRow, iCol] = image[int(transform_coord[0, 1]), int(transform_coord[0, 0])]
                # new_image[int(transform_coord[0, 1]), int(transform_coord[0, 0])] = image[iRow, iCol]

    # crop the transformed_image
    # image_centre = np.array([image.shape[1]/2, image.shape[0]/2])
    # image_centre = np.expand_dims(image_centre, axis=0)
    # transform_image_centre = transform_points(image_centre, transform)
    # new_start_row, new_start_col = int(transform_image_centre[0, 1] - image.shape[0]/2), int(transform_image_centre[0, 0] - image.shape[1]/2)
    # image_start = np.array([[0, 0]])
    # transform_image_start = transform_points(image_start, transform).astype(int)
    transformed_image = new_image[0:image.shape[0], 0:image.shape[1]]

    _, axs = plt.subplots(1, 2, figsize=(15, 10))
    axs[0].imshow(image)
    axs[0].set_title("Input Image")
    axs[1].imshow(transformed_image)
    axs[1].set_title("Transformed Image")
    plt.show()
    print('hello')
    return transformed_image


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = 'D:\\Dropbox\\RF_Warwick\\Projects\\HIMICO\\Dataset\\leuven\\single_cell_tumour_ihc_batch1\\single_thumbnails_level6'
    image = imread(os.path.join(path, 'B-1986096_B4_CDX2p_MUC2y_MUC5g_CD8dab.png'))
    fixed_image = image[500:1000, 300:800, 0]

    transform_matrix = np.array([[1, 0, -5], [0, 1, 5], [0, 0, 1]]).astype(float)
    moving_image = cv2.warpAffine(fixed_image, transform_matrix[0:-1], fixed_image.shape[:2][::-1])

    _, axs = plt.subplots(1, 2, figsize=(15, 10))
    axs[0].imshow(fixed_image)
    axs[0].set_title("Fixed Image")
    axs[1].imshow(moving_image)
    axs[1].set_title("Moving Image")
    plt.show()

    transformed_image = apply_transform(moving_image, inv(transform_matrix))

    _, axs = plt.subplots(1, 3, figsize=(15, 10))
    axs[0].imshow(fixed_image)
    axs[0].set_title("Fixed Image")
    axs[1].imshow(moving_image)
    axs[1].set_title("Moving Image")
    axs[2].imshow(transformed_image)
    axs[2].set_title("Transformed Image")
    plt.show()

