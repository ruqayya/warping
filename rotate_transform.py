from skimage.io import imread
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools
from numpy.linalg import inv
import cv2
import time
import utils


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

    start_time = time.time()
    transformed_image = utils.apply_transform(moving_image, inv(transform_matrix)).astype(int)
    end_time = time.time()
    total_time = end_time -start_time
    print('Time taken %0.2f seconds'%(total_time))

    registered_overlay = np.dstack((transformed_image, fixed_image, transformed_image))
    _, axs = plt.subplots(1, 3, figsize=(15, 10))
    axs[0].imshow(fixed_image, cmap='gray')
    axs[0].set_title("Fixed Image")
    axs[1].imshow(moving_image, cmap='gray')
    axs[1].set_title("Moving Image")
    axs[2].imshow(registered_overlay, cmap='gray')
    axs[2].set_title("Registered Overlay")
    plt.show()

