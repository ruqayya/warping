from skimage.io import imread
import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import itertools
from numpy.linalg import inv
import cv2
import time
import utils


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fixed_image = imread('fixed_img.png')

    transform_matrix = np.array([[1, 0, -5], [0, 1, 5], [0, 0, 1]]).astype(float)
    moving_image = cv2.warpAffine(fixed_image, transform_matrix[0:-1], fixed_image.shape[:2][::-1])

    # _, axs = plt.subplots(1, 2, figsize=(15, 10))
    # axs[0].imshow(fixed_image)
    # axs[0].set_title("Fixed Image")
    # axs[1].imshow(moving_image)
    # axs[1].set_title("Moving Image")
    # plt.show()

    start_time = time.time()
    transformed_image = utils.apply_transform(moving_image, inv(transform_matrix)).astype(int)
    end_time = time.time()
    total_time = end_time -start_time
    print('Time taken %0.2f seconds'%(total_time))

    registered_overlay = np.dstack((transformed_image, fixed_image, transformed_image))

    _, axs = plt.subplots(1, 3, figsize=(15, 10))
    axs[0].imshow(fixed_image)
    axs[0].set_title("Fixed Image")
    axs[1].imshow(moving_image)
    axs[1].set_title("Moving Image")
    axs[2].imshow(registered_overlay)
    axs[2].set_title("Registered Overlay")
    plt.show()

