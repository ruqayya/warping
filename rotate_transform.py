from skimage.io import imread
import numpy as np
import os
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

    transformed_image = new_image[0:image.shape[0], 0:image.shape[1]]

    _, axs = plt.subplots(1, 2, figsize=(15, 10))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Input Image")
    axs[1].imshow(transformed_image, cmap='gray')
    axs[1].set_title("Transformed Image")
    plt.show()

    return transformed_image


if __name__ == '__main__':
    fixed_image = imread('fixed_img.png')

    transform_matrix_90 = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).astype(float)
    # rotation_matrix_45 = np.array([[0.7071, 0.7071, 0], [-0.7071, 0.7071, 0], [0, 0, 1]]).astype(float)
    inverse_translation = np.array([[1, 0, -fixed_image.shape[1]/2], [0, 1, -fixed_image.shape[0]/2], [0, 0, 1]])
    forward_translation = np.array([[1, 0, fixed_image.shape[1]/2], [0, 1, fixed_image.shape[0]/2], [0, 0, 1]])
    translation_matrix = np.array([[1, 0, 50], [0, 1, 50], [0, 0, 1]]).astype(float)
    transform_matrix = forward_translation @ transform_matrix_90 @ inverse_translation
    transform_matrix = translation_matrix @ transform_matrix
    moving_image = cv2.warpAffine(fixed_image, transform_matrix[0:-1], fixed_image.shape[:2][::-1])

    # _, axs = plt.subplots(1, 2, figsize=(15, 10))
    # axs[0].imshow(fixed_image, cmap='gray')
    # axs[0].set_title("Fixed Image")
    # axs[1].imshow(moving_image, cmap='gray')
    # axs[1].set_title("Moving Image")
    # plt.show()

    transformed_image = apply_transform(moving_image, inv(transform_matrix)).astype(int)

    registered_overlay = np.dstack((transformed_image, fixed_image, transformed_image))
    _, axs = plt.subplots(1, 3, figsize=(15, 10))
    axs[0].imshow(fixed_image, cmap='gray')
    axs[0].set_title("Fixed Image")
    axs[1].imshow(moving_image, cmap='gray')
    axs[1].set_title("Moving Image")
    axs[2].imshow(registered_overlay, cmap='gray')
    axs[2].set_title("Registered Overlay")
    plt.show()

